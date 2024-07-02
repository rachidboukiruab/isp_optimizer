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
    """
    Node for loading datasets using VOC annotations format and publishing the paths of the images
    to be read by the Image Signal Processor (ISP).
    """

    def __init__(self):
        super().__init__("data_loader")

        # Declare parameters
        self.declare_parameter("images_folder_path", "/home/rachid/raw_dataset/test/")
        self.declare_parameter("batch_size", 1)
        self.declare_parameter("verbose_level", 3)

        # Initialize parameters
        self._images_folder_path = self.get_parameter("images_folder_path").value
        self._batch_size = self.get_parameter("batch_size").value
        self._verbose_level = self.get_parameter("verbose_level").value
        self._counter_images_sent_to_isp = 0

        # Load dataset dictionary
        self.dataset_dict = load_dataset(self._images_folder_path)

        # Create a service to trigger batch image processing
        self.process_batch_images_server_ = self.create_service(
            Trigger, "process_batch_images", self.callback_process_batch_images
        )

        # Create a publisher for raw image paths
        self.raw_image_path_publisher_ = self.create_publisher(
            String, "raw_image_path", self._batch_size
        )

        # Log initialization
        if self._verbose_level >= 2:
            self.get_logger().info("DataLoaderNode initialized.")
            self.get_logger().info("Node parameters:")
            self.get_logger().info(f"   - annotations_folder_path: {self._images_folder_path}")
            self.get_logger().info(f"   - batch_size: {self._batch_size}")

    def callback_process_batch_images(self, request, response):
        """
        Service callback for processing a batch of images.

        Args:
            request (TriggerRequest): The request object for the service (not used here).
            response (TriggerResponse): The response object to be sent back to the caller.

        Returns:
            TriggerResponse: Response with a success status.
        """
        start_time = time_ns()
        
        # Select a sample of images based on the batch size
        if len(self.dataset_dict) <= self._batch_size:
            sample_of_images = self.dataset_dict.keys()
        else:
            sample_of_images = random.sample(self.dataset_dict.keys(), self._batch_size)

        # Publish the paths of the sampled images
        self.publish_batch_images(sample_of_images)
        
        # Set the response as successful
        response.success = True
        loop_time_ms = (time_ns() - start_time) / 1000000.0
        
        # Log processing time
        if self._verbose_level >= 1:
            self.get_logger().info(
                f"All images has been sent to ISP. Processing time: {loop_time_ms:.3f} ms."
            )

        return response

    def publish_batch_images(self, sample_of_images):
        """
        Publish paths of the sampled images.

        Args:
            sample_of_images (List[str]): List of image names to be published.

        This function iterates over the list of sampled images and publishes their paths.
        """
        for image in sample_of_images:
            image_path = self.dataset_dict[image]["img_path"]
            self.publish_image_path(image, image_path)

        # Wait until all images are published
        while self._counter_images_sent_to_isp < self._batch_size:
            pass

        self._counter_images_sent_to_isp = 0

    def publish_image_path(self, image_name, image_path):
        """
        Publish the path of an image.

        Args:
            image_name (str): The name of the image.
            image_path (str): The path to the image.

        This function publishes the path of an image to the 'raw_image_path' topic.
        """
        msg = String()
        msg.data = image_path
        self.raw_image_path_publisher_.publish(msg)
        self._counter_images_sent_to_isp += 1
        
        # Log the published image information
        if self._verbose_level >= 3:
            remaining_images = self._batch_size - self._counter_images_sent_to_isp
            self.get_logger().info(
                f"RAW image '{image_name}.raw' published. Remaining {remaining_images} images."
            )


def main(args=None):
    """
    Main function for running the DataLoaderNode.

    Args:
        args (optional): Arguments for initializing the node.

    This function initializes the ROS node and keeps it spinning.
    """
    rclpy.init(args=args)
    node = DataLoaderNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
