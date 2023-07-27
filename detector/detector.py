import torch
import cv2
import numpy as np
from detector.utils import ResizeWithAspectRatio, get_frame_mean_IoU, get_frame_ap, \
    load_model, inference_onnx, scale_coords, ficosa_classes


class Detector:
    """ TODO"""
    def __init__(self, ficosa_model=False, model_path=''):
        """
        :param cfg: yacs.Config object, configurations about camera specs and module parameters
        """
        self.ficosa_model = ficosa_model
        # Load model
        if ficosa_model:
            self.model, self.imgsz = load_model(model_path, imgsz=640, conf_thres=0.5, device='cuda: 0')
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)

        self.class_colors = {
            "car": (128, 0, 128),  # Purple
            "person": (0, 0, 139),  # Dark red
            "motorbike": (255, 255, 0),  # Cyan
            "truck": (0, 139, 139)  # Yellow
        }

    def inference(self, img, visualize_img = False, gt_bboxes = [], classes = ["car", "truck"],
                  min_conf = 0.5, show_gt = True, save_img=False, out_path = ""):
        """
        Performs the object detection in a batch of images, evaluates the predictions and outputs the precision
        :param frames: path of the images that will be evaluated
        :param gt_labels: ground truth bounding boxes
        :param verbose: whether to print timing messages
        :param show_img: whether to show the image with detections or not.
        :param save_img: whether to save the image with detections or not.

        :return:
            precision: float value with the precision obtained after evaluating the predictions
        """
        # process frame with yolo
        if self.ficosa_model:
            img1, preds = inference_onnx(self.model, self.imgsz, img)

            img1 = np.ascontiguousarray(img1[0].transpose((1, 2, 0)))
            results = []
            for det in preds:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img1.shape, det[:, :4], img.shape).round()

                    for *xyxy, conf, cls in reversed(det):

                        if conf < min_conf or int(cls) in [6, 7]: continue

                        c = ficosa_classes(int(cls))

                        output_pred = [*[int(j.numpy()) for j in xyxy], float(conf.numpy()), c]
                        results.append(output_pred)
        else:
            results = self.model(img)

        if visualize_img or save_img:
            self.show_bounding_boxes(img, results, gt_bboxes, classes, min_conf, show_gt, visualize_img,
                                     save_img, out_path)

        iou, mAP, _ = self.evaluate_predictions(results, gt_bboxes, classes, min_conf)

        return iou, mAP

    def show_bounding_boxes(self, img, results, gt_bboxes, classes=["car"], min_conf=0.5, show_gt=True,
                            visualize_img=False, save_img=False, out_path=""):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.ficosa_model:
            bboxes = results
        else:
            bboxes = results.xyxy[0]  # Obtener las bounding boxes

        for bbox in bboxes:
            if self.ficosa_model:
                class_name = bbox[5]
            else:
                class_name = results.names[int(bbox[5])]

            if (class_name in classes) and (bbox[4] >= min_conf):
                x1, y1, x2, y2 = map(int, bbox[:4])  # bounding box coordinates
                label = f"{class_name}: {bbox[4]:.2f}"  # Class and confidence score

                color = self.class_colors.get(class_name, (0, 0, 0)) # black if class not exists

                # Draw bounding box in the image
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

        if show_gt:
            for gt_bbox in gt_bboxes:
                x1, y1, x2, y2 = map(int, gt_bbox[:4])  # bounding box coordinates
                # Draw bounding box in the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if visualize_img:
            # Show image with bounding boxes
            resize = ResizeWithAspectRatio(img, width=1280)  # Resize by width
            cv2.imshow("Detections", resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_img:
            cv2.imwrite(out_path, img)

    def evaluate_predictions(self, results, gt_bboxes, classes=["car", 'truck'], min_conf=0.5):
        if self.ficosa_model:
            det_ann = results
        else:
            det_ann = results.xyxy[0]

        det_bboxes = []
        filtered_gt_bboxes = []
        n = 0
        sum_iou = 0
        sum_mAP = 0
        metrics_dict = {}
        for cls in classes:
            for det_bbox in det_ann:
                if self.ficosa_model:
                    class_name = det_bbox[5]
                else:
                    class_name = results.names[int(det_bbox[5])]

                if class_name in cls:
                    det_bboxes.append([int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2]), int(det_bbox[3]), float(det_bbox[4])])
            for gt_bbox in gt_bboxes:
                class_name = gt_bbox[4]
                if class_name in cls:
                    filtered_gt_bboxes.append([int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[2]), int(gt_bbox[3])])

            cls_iou = get_frame_mean_IoU(filtered_gt_bboxes, det_bboxes)
            cls_ap = get_frame_ap(filtered_gt_bboxes, det_bboxes, confidence=True, n=10, th=min_conf)

            n += 1
            sum_iou += cls_iou
            sum_mAP += cls_ap

            metrics_dict[cls] = {'IoU': cls_iou, 'AP': cls_ap}

        mIoU = sum_iou / n
        mAP = sum_mAP / n

        return mIoU, mAP, metrics_dict

