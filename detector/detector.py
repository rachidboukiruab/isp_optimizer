import torch
import cv2
import numpy as np
import time
from detector.utils import ResizeWithAspectRatio, get_frame_mean_IoU, get_frame_ap, \
    load_model, inference_onnx, scale_coords, ficosa_classes, get_coco_name_from_id


class Detector:
    """ Image detector and evaluation using YOLOv5n pretrained model as default or Ficosa custom model"""
    def __init__(self, ficosa_model=False, model_path=''):
        """
        :param ficosa_model: Boolean. If is true the detector will use Ficosa custom model for inference. If is false
                            the detector will use YOLOv5n pretrained model for inference. Default value False.
        :param model_path: String. If ficosa_model is true we will load the custom model placed in the path defined in
                            this variable.
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

    def inference(self, batch_imgs, batch_bboxes, visualize_img = False, classes = ["car"],
                  min_conf = 0.5, show_gt = True, save_img=False, out_path = "", verbose=False):
        """
        :param batch_imgs: List. Batch of processed images where we want to perform the inference.
        :param batch_bboxes: List. The ground truth of the batch_imgs.
        :param visualize_img: Boolean. If is true will be shown the processed images with the detections drawn in them.
                            Default value False.
        :param classes: List. The classes we want to perform the inference and evaluate.
                        Default value ["car"].
        :param min_conf: Float. Confidence threshold of the detected bounding boxes. Default value 0.5.
        :param show_gt: Boolean. If is true will be shown the ground truth bboxes drawn in the images. Default value True.
        :param save_img: Boolean. If is true the images with the detections will be saved in the specified out_path folder.
                        Default value False.
        :param out_path: String. The output folder where the images will be saved. Default value null.
        :param verbose: Boolean. If is true will be printed in the terminal the computation times of each process.
                        Default value False.
        :return: Dictionary with the IoU, AP, Precision, and Recall for each class and processed image.
        """
        if verbose:
            print("Starting inference...")
            begin = time.time_ns()
        if self.ficosa_model:
            batch_results = []
            for i in range(len(batch_imgs)):
                img1, preds = inference_onnx(self.model, self.imgsz, batch_imgs[i])

                img1 = np.ascontiguousarray(img1[0].transpose((1, 2, 0)))
                results = []
                for det in preds:
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img1.shape, det[:, :4], batch_imgs[i].shape).round()

                        for *xyxy, conf, cls in reversed(det):

                            if conf < min_conf or int(cls) in [6, 7]: continue

                            c = ficosa_classes(int(cls))

                            output_pred = [*[int(j.numpy()) for j in xyxy], float(conf.numpy()), c]
                            results.append(output_pred)
                batch_results.append(results)

        else:
            # process frame with yolo
            batch_results = self.model(batch_imgs)
        if verbose:
            print(f"Inference time: {(time.time_ns() - begin) / 1000000} ms")
            print("Starting evaluation...")
            begin = time.time_ns()
        iter_metrics_dict = {}
        for cls in classes:
            iter_metrics_dict[cls] = {'IoU': [], 'AP': [], 'Precision': [], 'Recall': []}

        for i in range(len(batch_imgs)):
            if self.ficosa_model:
                if visualize_img or save_img:
                    self.show_bounding_boxes(batch_imgs[i], batch_results[i], batch_bboxes[i], classes, min_conf,
                                             show_gt, visualize_img, save_img, out_path)
                metrics_dict = self.evaluate_predictions(batch_results[i], batch_bboxes[i], classes, min_conf)
            else:
                if visualize_img or save_img:
                    self.show_bounding_boxes(batch_imgs[i], batch_results.xyxy[i], batch_bboxes[i], classes, min_conf,
                                             show_gt, visualize_img, save_img, out_path)
                metrics_dict = self.evaluate_predictions(batch_results.xyxy[i], batch_bboxes[i], classes, min_conf)

            # metrics_dict struct:
            # metrics_dict[cls] = {'IoU': cls_iou, 'AP': cls_ap, 'Precision': cls_prec, 'Recall': cls_rec}
            for cls in classes:
                if cls in metrics_dict.keys():
                    iter_metrics_dict[cls]['IoU'].append(metrics_dict[cls]['IoU'])
                    iter_metrics_dict[cls]['AP'].append(metrics_dict[cls]['AP'])
                    iter_metrics_dict[cls]['Precision'].append(metrics_dict[cls]['Precision'])
                    iter_metrics_dict[cls]['Recall'].append(metrics_dict[cls]['Recall'])
        if verbose:
            print(f"Evaluation time: {(time.time_ns() - begin) / 1000000} ms")
        return iter_metrics_dict

    def show_bounding_boxes(self, img, bboxes, gt_bboxes, classes=["car"], min_conf=0.5, show_gt=True,
                            visualize_img=False, save_img=False, out_path=""):
        """
        :param img: Image array. Processed image.
        :param bboxes: List. Bounding boxes detected in the processed image.
        :param gt_bboxes: List. Bounding boxes ground truth of the processed image.
        :param classes: List. The classes we want to draw in the image.
                        Default value ["car"].
        :param min_conf: Float. Confidence threshold of the detected bounding boxes. Default value 0.5.
        :param show_gt: Boolean. If is true will be shown the ground truth bboxes drawn in the image. Default value True.
        :param visualize_img: Boolean. If is true will be shown the processed image with the detections drawn in it.
                            Default value False.
        :param save_img: Boolean. If is true the image with the detections will be saved in the specified out_path folder.
                        Default value False.
        :param out_path: String. The output folder where the image will be saved. Default value null.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for bbox in bboxes:
            if self.ficosa_model:
                class_name = bbox[5]
            else:
                class_name = get_coco_name_from_id(int(bbox[5]))

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

    def evaluate_predictions(self, det_ann, gt_bboxes, classes=["car"], min_conf=0.5):
        """
        :param det_ann: List. Bounding boxes detected in the processed image.
        :param gt_bboxes: List. Bounding boxes ground truth of the processed image.
        :param classes: List. The classes we want to draw in the image.
                        Default value ["car"].
        :param min_conf: Float. Confidence threshold of the detected bounding boxes. Default value 0.5.
        :return: Dictionary with the following structure
                metrics_dict[string(cls)] = {'IoU': float(cls_iou), 'AP': float(cls_ap), 'Precision': float(cls_prec),
                'Recall': float(cls_rec)}
        """
        metrics_dict = {}
        for cls in classes:
            det_bboxes = []
            filtered_gt_bboxes = []
            for det_bbox in det_ann:
                if self.ficosa_model:
                    class_name = det_bbox[5]
                else:
                    class_name = get_coco_name_from_id(int(det_bbox[5]))

                if class_name in cls:
                    det_bboxes.append([int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2]), int(det_bbox[3]),
                                       float(det_bbox[4])])
            for gt_bbox in gt_bboxes:
                class_name = ficosa_classes(gt_bbox[4])
                if class_name in cls:
                    filtered_gt_bboxes.append([int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[2]), int(gt_bbox[3])])

            cls_iou = get_frame_mean_IoU(filtered_gt_bboxes, det_bboxes)
            cls_ap, cls_prec, cls_rec = get_frame_ap(filtered_gt_bboxes, det_bboxes, confidence=True, n=10, th=min_conf)

            if cls_ap is not None:
                metrics_dict[cls] = {'IoU': cls_iou, 'AP': cls_ap, 'Precision': cls_prec, 'Recall': cls_rec}

        return metrics_dict

