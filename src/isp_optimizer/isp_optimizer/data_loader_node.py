#!/usr/bin/env python3
# Standard library imports
import random
from time import time_ns

# External package imports
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String

# Local package imports
from .utils import load_dataset


class DataLoaderNode(Node):
    """Node for loading datasets using VOC annotations format and publishing the paths of the images to be read by the ISP."""

    def __init__(self):
        super().__init__("data_loader")

        # Parameters
        self.declare_parameter("images_folder_path", "/home/rachid/raw_dataset/test/")
        self.declare_parameter("batch_size", 1)
        #Verbose levels: 
        #    - 3 for debug logs
        #    - 2 for relevant user information
        #    - 1 for only warning and error logs
        #    - 0 for only error logs
        self.declare_parameter("verbose_level", 3)

        # Initialize parameters
        self._images_folder_path = self.get_parameter("images_folder_path").value
        self._batch_size = self.get_parameter("batch_size").value
        self._counter_images_sent_to_isp = 0
        self._verbose_level = self.get_parameter("verbose_level").value

        # Load dataset
        self.dataset_dict = load_dataset(self._images_folder_path)

        # Service topic to start processing a batch of images
        self.process_batch_images_server_ = self.create_service(
            Trigger, "process_batch_images", self.callback_process_batch_images
        )

        # Publishers and subscribers of the HDR-ISP
        self.raw_image_path_publisher_ = self.create_publisher(
            String, "raw_image_path", self._batch_size
        )

        # Log initialization
        if self._verbose_level >= 2:
            self.get_logger().info("DataLoaderNode initialized.")
            self.get_logger().info("Node parameters:")
            self.get_logger().info(
                f"   - annotations_folder_path: {self._images_folder_path}"
            )
            self.get_logger().info(f"   - batch_size: {self._batch_size}")

    def callback_process_batch_images(self, request, response):
        """Publish the path of the raw image."""
        start_time = time_ns()
        if len(self.dataset_dict) <= self._batch_size:
            sample_of_images = self.dataset_dict.keys()
        else:
            sample_of_images = random.sample(self.dataset_dict.keys(), self._batch_size)

        self.publish_batch_images(sample_of_images)
        response.success = True
        loop_time_ms = (time_ns() - start_time) / 1000000.0
        if self._verbose_level >= 1:
            self.get_logger().info(
                f"All images has been sent to ISP. Processing time: {loop_time_ms:.3f} ms."
            )

        return response

    def publish_batch_images(self, sample_of_images):
        for image in sample_of_images:
            image_path = self.dataset_dict[image]["img_path"]
            self.publish_image_path(image, image_path)

        while self._counter_images_sent_to_isp < self._batch_size:
            pass

        self._counter_images_sent_to_isp = 0

    def publish_image_path(self, image_name, image_path):
        """Publish path of an image."""
        msg = String()
        msg.data = image_path
        self.raw_image_path_publisher_.publish(msg)
        self._counter_images_sent_to_isp += 1
        if self._verbose_level >= 3:
            self.get_logger().info(
                f"RAW image with name {image_name}.raw published. Remaining {self._batch_size - self._counter_images_sent_to_isp} images."
            )


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = DataLoaderNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
