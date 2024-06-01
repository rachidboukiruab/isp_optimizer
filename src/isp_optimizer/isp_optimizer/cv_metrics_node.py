#!/usr/bin/env python3
from time import time_ns
# External package imports
import rclpy
from rclpy.node import Node
import torch
from torchmetrics.detection import MeanAveragePrecision

# Local package imports
from .utils import load_dataset
from isp_optimizer_interfaces.msg import ComputerVisionMetrics, ListBoundingBox
from isp_optimizer_interfaces.srv import GetMetricResults


class CvMetricsNode(Node):
    """mAP and mAR extractor from a set of pairs of bounding boxes"""

    def __init__(self):
        super().__init__("cv_metrics")

        # Parameters
        self.declare_parameter("annotations_folder_path", "/home/rachid/raw_dataset/test/")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("classes_list", ["car"])
        self.declare_parameter("batch_size", 1)
        #Verbose levels: 
        #    - 3 for debug logs
        #    - 2 for relevant user information
        #    - 1 for only warning and error logs
        #    - 0 for only error logs
        self.declare_parameter("verbose_level", 3)

        # Initialize parameters
        self._annotations_folder_path = self.get_parameter("annotations_folder_path").value
        self._confidence_threshold = self.get_parameter("confidence_threshold").value
        self._classes_list = self.get_parameter("classes_list").value
        self._batch_size = self.get_parameter("batch_size").value
        self._verbose_level = self.get_parameter("verbose_level").value
        self._counter_results_received_from_inference = 0
        self._classes_dict_map = {
            self._classes_list[class_id]: class_id
            for class_id in range(len(self._classes_list))
        }
        self._predicted_bounding_boxes_list = []
        self._groundtruth_bounding_boxes_list = []
        self._mean_average_precision = 0.0
        self._mean_average_recall = 0.0
        self._results_are_available = False
        self.start_time = 0
        
        # Load dataset
        self.dataset_dict = load_dataset(self._annotations_folder_path)

        # Subscriber where to receive predicted bounding boxes
        self.evaluate_inference_results_subscriber_ = self.create_subscription(
            ListBoundingBox, "run_evaluation", self.callback_evaluate_inference_results, self._batch_size
        )

        self._send_metric_results_if_available = self.create_service(
            GetMetricResults,
            "cv_metric_results",
            self.callback_send_metric_results_if_available,
        )

        # Log initialization
        if self._verbose_level >= 2:
            self.get_logger().info("Computer Vision Metrics node initialized.")

    def callback_evaluate_inference_results(self, msg):
        if self._counter_results_received_from_inference == 0:
            self.start_time = time_ns()
        image_path = msg.frame_id
        image_name = (image_path.split("/")[-1]).split(".")[0]        
        groundtruth_bounding_boxes = self.convert_list_to_torchmetrics(
            self.dataset_dict[image_name]["bboxes"], confidence_scores=False
        )
        predicted_bounding_boxes = self.convert_boundingbox_to_torchmetrics(
            msg.bounding_boxes, confidence_scores=True
        )
        # Add results to their respective lists
        self._predicted_bounding_boxes_list.append(predicted_bounding_boxes)
        self._groundtruth_bounding_boxes_list.append(groundtruth_bounding_boxes)
        self._counter_results_received_from_inference += 1
        if self._verbose_level >= 3:
                self.get_logger().info(f"Messages received = {str(self._counter_results_received_from_inference)}")
        
        # All results are received        
        if self._counter_results_received_from_inference >= self._batch_size:
            time_receive_bboxes = (time_ns() - self.start_time) / 1000000.
            if self._verbose_level >= 1:
                self.get_logger().info(f"All bounding boxes are received. Processing time: {time_receive_bboxes:.3f} ms.")
            self.start_time = time_ns()
            # Calculate mAP
            map = MeanAveragePrecision(class_metrics=True)
            map.update(
                self._predicted_bounding_boxes_list,
                self._groundtruth_bounding_boxes_list,
            )
            map_result = map.compute()
            self._mean_average_precision = map_result["map_50"].item()
            self._mean_average_recall = map_result["mar_100"].item()
            # reset counter, bounding boxes lists and enable the results_are_available flag
            self._counter_results_received_from_inference = 0
            self._predicted_bounding_boxes_list = []
            self._groundtruth_bounding_boxes_list = []
            self._results_are_available = True
            loop_time_ms = (time_ns() - self.start_time) / 1000000.
            if self._verbose_level >= 1:
                self.get_logger().info(f"Results are available. mAP50 = {str(self._mean_average_precision)}. Processing time: {loop_time_ms:.3f} ms.")

    def callback_send_metric_results_if_available(self, request, response):
        if self._results_are_available:
            result_msg = ComputerVisionMetrics()
            result_msg.mean_average_precision = self._mean_average_precision
            result_msg.mean_average_recall = self._mean_average_recall
            response.result = result_msg
            response.success = True
            if self._verbose_level >= 3:
                self.get_logger().info(
                    "Sent response mAP_50 = "
                    + str(result_msg.mean_average_precision)
                    + "; mAR_100 = "
                    + str(result_msg.mean_average_recall)
                )
            self._results_are_available = False
        else:
            response.success = False
            if self._verbose_level >= 4:
                self.get_logger().info(
                    "Still don't have the results available."
                )

        return response

    def convert_boundingbox_to_torchmetrics(
        self, bounding_boxes, confidence_scores=False
    ):
        boxes_list = []
        scores_list = []
        labels_list = []
        for bounding_box in bounding_boxes:
            if bounding_box.class_name in self._classes_list:
                boxes_list.append(
                    [
                        bounding_box.top_left_x,
                        bounding_box.top_left_y,
                        bounding_box.bottom_right_x,
                        bounding_box.bottom_right_y,
                    ]
                )
                labels_list.append(self._classes_dict_map[bounding_box.class_name])
                if confidence_scores:
                    scores_list.append(bounding_box.confidence_score)

        torchmetrics_dict = {
            "boxes": torch.tensor(boxes_list),
            "scores": torch.tensor(scores_list) if confidence_scores else None,
            "labels": torch.tensor(labels_list),
        }

        return torchmetrics_dict
    
    def convert_list_to_torchmetrics(
        self, bounding_boxes, confidence_scores=False
    ):
        boxes_list = []
        scores_list = []
        labels_list = []
        for bounding_box in bounding_boxes:
            if bounding_box[5] in self._classes_list:
                boxes_list.append(bounding_box[:4])
                labels_list.append(self._classes_dict_map[bounding_box[5]])
                if confidence_scores:
                    scores_list.append(bounding_box[4])

        torchmetrics_dict = {
            "boxes": torch.tensor(boxes_list),
            "scores": torch.tensor(scores_list) if confidence_scores else None,
            "labels": torch.tensor(labels_list),
        }

        return torchmetrics_dict


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = CvMetricsNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
