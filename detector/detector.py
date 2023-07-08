import torch
import cv2
from detector.utils import ResizeWithAspectRatio, get_frame_mean_IoU, get_frame_ap


class Detector:
    """ Core fast-openISP pipeline """
    def __init__(self):
        """
        :param cfg: yacs.Config object, configurations about camera specs and module parameters
        """
        # Load model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)

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
        results = self.model(img)

        if visualize_img or save_img:
            self.show_bounding_boxes(img, results, gt_bboxes, classes, min_conf, show_gt, visualize_img,
                                     save_img, out_path)

        iou, mAP = self.evaluate_predictions(results, gt_bboxes, classes, min_conf)

        return iou, mAP

    def show_bounding_boxes(self, img, results, gt_bboxes, classes=["car"], min_conf=0.5, show_gt=True,
                            visualize_img=False, save_img=False, out_path=""):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bboxes = results.xyxy[0]  # Obtener las bounding boxes

        for bbox in bboxes:
            if (results.names[int(bbox[5])] in classes) and (bbox[4] >= min_conf):
                x1, y1, x2, y2 = map(int, bbox[:4])  # bounding box coordinates
                label = f"{bbox[4]:.2f}"  # confidence score

                # Draw bounding box in the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if show_gt:
            for gt_bbox in gt_bboxes:
                x1, y1, x2, y2 = map(int, gt_bbox[:4])  # bounding box coordinates
                # Draw bounding box in the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if visualize_img:
            # Show image with bounding boxes
            resize = ResizeWithAspectRatio(img, width=1280)  # Resize by width
            cv2.imshow("Detections", resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_img:
            cv2.imwrite(out_path, img)

    def evaluate_predictions(self, results, gt_bboxes, classes=["car", 'truck'], min_conf=0.5):

        det_ann = results.xyxy[0]
        det_bboxes = []
        for det_bbox in det_ann:
            if (results.names[int(det_bbox[5])] in classes):
                det_bboxes.append([int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2]), int(det_bbox[3]), float(det_bbox[4])])

        iou = get_frame_mean_IoU(gt_bboxes, det_bboxes)
        mAP = get_frame_ap(gt_bboxes, det_bboxes, confidence=True, n=10, th=min_conf)

        return iou, mAP

