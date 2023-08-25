import cv2
import numpy as np
import random
import onnxruntime
import torch
import math
import time
import os


def get_coco_name_from_id(num):
    id_to_name = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorbike',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }

    return id_to_name[num]


def ficosa_classes(num):
    id_to_name = {
        0: 'car',
        1: 'truck',
        2: 'bicycle',
        3: 'person',
        4: 'motorbike',
        5: 'bus',
        6: 'traffic_sign',
        7: 'traffic_light'
    }

    return id_to_name[num]

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def get_IoU(bbox_a, bbox_b):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Args:
    - box_A bounding box a
    - boxb: bounding box b

    Returns:
    - The IoU value between the two bounding boxes.
    """

    # Get the coordinates of the bounding boxes
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b

    # Compute the area of the intersection rectangle
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Compute the area of both bounding boxes
    bbox_a_area = (x2_a - x1_a + 1) * (y2_a - y1_a + 1)
    bbox_b_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    # Compute the IoU
    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)

    return iou

def get_frame_IoU(gt_bboxes, det_bboxes):
    used_gt_idxs = set()  # keep track of which ground truth boxes have already been used
    frame_iou = []
    for det_bbox in det_bboxes:
        max_iou = 0
        max_gt_idx = None
        for i, gt_bbox in enumerate(gt_bboxes):
            if i in used_gt_idxs:
                continue  # skip ground truth boxes that have already been used
            iou = get_IoU(gt_bbox, det_bbox[:4])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = i
        if max_gt_idx is not None:
            used_gt_idxs.add(max_gt_idx)
            frame_iou.append(max_iou)
        else:  # False Positives (aka pred boxes that do not intersect with any gt box)
            frame_iou.append(0)
    # False negative: check if there are any ground truth boxes that were not used
    for i, gt_bbox in enumerate(gt_bboxes):
        if i not in used_gt_idxs:
            frame_iou.append(0)
    return frame_iou


def get_frame_mean_IoU(gt_bboxes, det_bboxes):
    if len(gt_bboxes) == 0:
        if len(det_bboxes) == 0:
            return 1.0
        else:
            return 0.
    return np.mean(get_frame_IoU(gt_bboxes, det_bboxes))


def ap_voc(frame_iou, total_det, total_gt, th):
    """
    Computes the Average Precision (AP) in a frame according to Pascal Visual Object Classes (VOC) Challenge.
    Args:
    - frame_iou: list with the IoU results of each ground truth bbox
    - total_det: int defining the number of bounding boxes detected
    - th: float defining a threshold of IoU metric. If IoU is higher than th,
          then the detected bb is a TP, otherwise is a FP.

    Returns:
    - The AP value of the bb detected.
    """
    # Define each detection if it is a true positive or a false positive
    tp = np.zeros(total_det)
    fp = np.zeros(total_det)

    for i in range(total_det):
        if frame_iou[i] > th:
            tp[i] = 1
        else:
            fp[i] = 1

    # Tabulate the cumulative sum of the true and false positives
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    # Compute Precision and Recall
    precision = tp / np.maximum((tp + fp), np.finfo(
        np.float64).eps)  # cumulative true positives / cumulative true positive + cumulative false positives
    recall = tp / float(total_gt)  # cumulative true positives / total ground truths

    pr = precision[-1]
    rc = recall[-1]
    if total_det < total_gt:
        precision = np.append(precision, 0.0)
        recall = np.append(recall, 1.0)

    # AP measurement according to the equations 1 and 2 in page 11 of
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    ap = 0.0
    for r in np.arange(0.0, 1.1, 0.1):
        if any(recall >= r):
            max_precision = np.max(precision[recall >= r])
            ap = ap + max_precision

    ap = ap / 11.0
    return ap, pr, rc


def get_frame_ap(gt_bboxes, det_bboxes, confidence=False, n=10, th=0.5):
    """
    Computes the Average Precision (AP) in a frame according to Pascal Visual Object Classes (VOC) Challenge.
    Args:
    - gt_bboxes: dictionary of ground truth bounding boxes
    - det_bboxes: dictionary of detected bounding boxes
    - confidence: True if we have the confidence score
    - n: Number of random sorted sets.
    - th: float defining a threshold of IoU metric. If IoU is higher than th,
          then the detected bb is a TP, otherwise is a FP.

    Returns:
    - The AP value of the bb detected in the frame.
    """
    total_gt = len(gt_bboxes)  # Number of bboxes in the ground truth
    total_det = len(det_bboxes)  # Number of bboxes in the predictions

    if total_gt == 0:
        # if we don't have any ground truth in the frame and also we don't have any prediction, we assume it's corret and we skip the frame
        if total_det == 0:
            return None, None, None
        # if we don't have any ground truth in the frame but we have predictions, we assume they are false positives and therefore the mAP is equal to 0
        else:
            return 0., 0., 0.
    if total_det == 0 and total_gt > 0:
        return 0., 0., 0.

    ap = 0.
    if confidence:
        # sort det_bboxes by confidence score in descending order
        det_bboxes.sort(reverse=True, key=lambda x: x[4])

        # Calculate the IoU of each detected bbox.
        frame_iou = get_frame_IoU(gt_bboxes, det_bboxes)[:total_det]

        #  Compute the AP
        ap, precision, recall = ap_voc(frame_iou, total_det, total_gt, th)
    else:
        # Generate N random sorted lists of the detections and compute the AP in each one
        ap_list = []
        precision_list = []
        recall_list = []
        for i in range(n):
            # sort randomly the det_bboxes
            random.shuffle(det_bboxes)

            # Calculate the IoU of each detected bbox.
            frame_iou = get_frame_IoU(gt_bboxes, det_bboxes)[:total_det]

            #  Compute the AP
            ap, precision, recall = ap_voc(frame_iou, total_det, total_gt, th)
            ap_list.append(ap)
            precision_list.append(precision)
            recall_list.append(recall)

        # Do the average of the computed APs
        ap = np.mean(ap_list)
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)

    return ap, precision, recall


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    #s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    s = ""
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    print(cuda)
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32, floor=0):
    # Verify img_size is a multiple of stride s
    new_size = max(make_divisible(img_size, int(s)), floor)  # ceil gs-multiple
    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def load_model(weights, imgsz, conf_thres, device):
    device = select_device(device)

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    assert w.endswith('.onnx')
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    # check_requirements(('onnx', 'onnxruntime'))

    session = onnxruntime.InferenceSession(w, None, providers=['CUDAExecutionProvider'])  # , 'CPUExecutionProvider'])
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    return session, imgsz


def inference_onnx(session, imgsz, img):
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    img = img.astype('float32')
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Padded resize
    img = letterbox(img, (imgsz, imgsz), stride=stride, scaleFill=True, auto=False)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    begin = time.time_ns()
    pred = torch.tensor(np.array(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img})))
    #print(f"Inference time: {(time.time_ns() - begin) / 1000000} ms")

    return img, pred


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        #gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        gain = (img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])# gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain[1]) / 2, (img1_shape[0] - img0_shape[0] * gain[0]) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [0, 2]] /= gain[1]
    coords[:, [1, 3]] /= gain[0]
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
