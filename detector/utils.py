import cv2
import numpy as np
import random

def get_coco_name_from_id(num):
    id_to_name = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
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
        return None
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
    return ap


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
            return None
        # if we don't have any ground truth in the frame but we have predictions, we assume they are false positives and therefore the mAP is equal to 0
        else:
            return 0.

    ap = 0.
    if confidence:
        # sort det_bboxes by confidence score in descending order
        det_bboxes.sort(reverse=True, key=lambda x: x[4])

        # Calculate the IoU of each detected bbox.
        frame_iou = get_frame_IoU(gt_bboxes, det_bboxes)[:total_det]

        #  Compute the AP
        ap = ap_voc(frame_iou, total_det, total_gt, th)
    else:
        # Generate N random sorted lists of the detections and compute the AP in each one
        ap_list = []
        for i in range(n):
            # sort randomly the det_bboxes
            random.shuffle(det_bboxes)

            # Calculate the IoU of each detected bbox.
            frame_iou = get_frame_IoU(gt_bboxes, det_bboxes)[:total_det]

            #  Compute the AP
            ap_list.append(ap_voc(frame_iou, total_det, total_gt, th))

        # Do the average of the computed APs
        ap = np.mean(ap_list)

    return ap