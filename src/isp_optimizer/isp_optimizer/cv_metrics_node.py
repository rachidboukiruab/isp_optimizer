#!/usr/bin/env python3

# External package imports
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from torchmetrics.detection import MeanAveragePrecision


# Local package imports
from isp_optimizer_interfaces.srv import RunEvaluation
from isp_optimizer_interfaces.msg import ComputerVisionMetrics


class CVMetricsNode(Node):
    """mAP and mAR extractor from a set of pairs of bounding boxes"""

    def __init__(self):
        super().__init__("cv_metrics")

        # Parameters
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("classes_list", ["car"])

        # Initialize parameters
        self._confidence_threshold = self.get_parameter("confidence_threshold").value
        self._classes_list = self.get_parameter("classes_list").value
        self._classes_dict_map = {
            self._classes_list[class_id]: class_id
            for class_id in range(len(self._classes_list))
        }
        self._map = MeanAveragePrecision()

        # Server Initializer
        self.evaluate_inference_results_server_ = self.create_service(
            RunEvaluation, "run_evaluation", self.callback_evaluate_inference_results
        )

        # Publisher initializer
        self.cv_metrics_publisher_ = self.create_publisher(
            ComputerVisionMetrics, "cv_metrics_result", 10
        )

        # Log initialization
        self.get_logger().info("Computer Vision Metrics node initialized.")

    def callback_evaluate_inference_results(self, request, response):
        predicted_bounding_boxes = self.convert_boundingbox_to_torchmetrics(
            request.predicted_bounding_boxes, confidence_scores=True
        )
        groundtruth_bounding_boxes = self.convert_boundingbox_to_torchmetrics(
            request.groundtruth_bounding_boxes, confidence_scores=False
        )

        # Calculate mAP
        self._map.update(predicted_bounding_boxes, groundtruth_bounding_boxes)
        map_result = self._map.compute()
        result_msg = ComputerVisionMetrics()
        result_msg.mean_average_precision = map_result["map"].item()
        result_msg.mean_average_recall = 0.0
        response.result = result_msg
        self.publish_map(result_msg)
        response.success = True
        self.get_logger().info(" Sent response " + str(result_msg.mean_average_precision))
        return response
    
    def publish_map(self, result_msg):
        self.cv_metrics_publisher_.publish(result_msg)
        self.get_logger().info("mAP and mAR published in cv_metrics_result topic.")

    def convert_boundingbox_to_torchmetrics(self, bounding_boxes, confidence_scores=False):
        boxes_list = []
        scores_list = []
        labels_list = []
        for bounding_box in bounding_boxes:
            if bounding_box.class_name in self._classes_list:
                boxes_list.append([bounding_box.top_left_x,
                                                      bounding_box.top_left_y,
                                                      bounding_box.bottom_right_x,
                                                      bounding_box.bottom_right_y,
                                                      ])
                labels_list.append(self._classes_dict_map[bounding_box.class_name])
                if confidence_scores:
                    scores_list.append(bounding_box.confidence_score)
                    
        torchmetrics_list = [{
            "boxes": torch.tensor(boxes_list),
            "scores": torch.tensor(scores_list) if confidence_scores else None,
            "labels": torch.tensor(labels_list)
        }]
                    
        return torchmetrics_list


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = CVMetricsNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
