#!/usr/bin/env python3

# Standard library imports
import os
import random

# External package imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from pylabel import importer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

# Local package imports
from isp_optimizer_interfaces.srv import RunInference
from isp_optimizer_interfaces.msg import BoundingBox


class DataLoaderNode(Node):
    """Node for loading datasets using VOC annotations format and publishing images."""

    def __init__(self):
        super().__init__("data_loader")

        # Parameters
        self.declare_parameter("annotations_folder_path", "/home/rachid/raw_dataset/test/")
        self.declare_parameter("classes_list", ["car"])
        self.declare_parameter("batch_size", 10)

        # Initialize parameters
        self._annotations_folder_path = self.get_parameter("annotations_folder_path").value
        self._classes_list = self.get_parameter("classes_list").value
        self._batch_size = self.get_parameter("batch_size").value
        self._counter_images_sent_to_inference = 0
        
        # Load dataset
        self.dataset_dict = self.load_dataset()
        
        # Service topic to start processing a batch of images
        self.process_batch_images_server_ = self.create_service(
            Trigger, "process_batch_images", self.callback_process_batch_images, callback_group=MutuallyExclusiveCallbackGroup())

        # Publishers and subscribers of the HDR-ISP
        self.raw_image_publisher_ = self.create_publisher(Image, "raw_image", 10)
        self.rgb_image_subscriber_ = self.create_subscription(
            Image, "rgb_image", self.callback_rgb_image, 10, callback_group=ReentrantCallbackGroup()
        )
        
        # Log initialization
        self.get_logger().info("DataLoaderNode initialized.")
        self.get_logger().info("Node parameters:")
        self.get_logger().info(f"   - annotations_folder_path: {self._annotations_folder_path}")
        self.get_logger().info(f"   - classes_list: {self._classes_list}")
        self.get_logger().info(f"   - batch_size: {self._batch_size}")

    def load_dataset(self):
        """
        Load dataset and preprocess it.
        Refactor the dataset to a dict with the image path of each image and corresponding bounding
        boxes. Example: 
        {'000': {'img_path': '/home/rachid/raw_dataset/test/000.raw', 
                'bboxes': [[top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                            confidence,
                            class_name]]}}
        """
        dataset_dataframe = importer.ImportVOC(path=self._annotations_folder_path)
        return self.convert_dataframe_to_dict(dataset_dataframe)

    def convert_dataframe_to_dict(self, dataframe):
        """
        Convert dataset dataframe to a dictionary.

        Args:
            dataframe: Pandas DataFrame generated by pylabel.importer object.

        Returns:
            Dictionary with image names as keys and corresponding paths and bounding boxes as values.
        """
        dataset_dict = {}

        # Filter the classes of interest
        dataframe_filtered_by_class = dataframe.df.query("cat_name in @self._classes_list")

        img_id_list = dataframe_filtered_by_class["img_id"].unique().tolist()

        for img_id in img_id_list:
            rows = dataframe_filtered_by_class[dataframe_filtered_by_class["img_id"] == img_id]
            img_name = rows["img_filename"].values[0]
            img_path = os.path.join(self._annotations_folder_path, img_name)
            bboxes = []
            for _, row in rows.iterrows():
                top_left_x = int(row["ann_bbox_xmin"])
                top_left_y = int(row["ann_bbox_ymin"])
                bottom_right_x = int(row["ann_bbox_xmax"])
                bottom_right_y = int(row["ann_bbox_ymax"])
                confidence = 1.0
                class_name = row["cat_name"]
                bboxes.append(
                    [
                        top_left_x,
                        top_left_y,
                        bottom_right_x,
                        bottom_right_y,
                        confidence,
                        class_name
                    ]
                )

            dataset_dict[img_name.split(".")[0]] = {
                "img_path": img_path,
                "bboxes": bboxes,
            }

        return dataset_dict


    def callback_process_batch_images(self, request, response):
        """Publish images from the dataset."""
        if len(self.dataset_dict) <= self._batch_size:
            sample_of_images = self.dataset_dict.keys()
        else:
            sample_of_images = random.sample(self.dataset_dict.keys(), self._batch_size)
        
        self.publish_batch_images(sample_of_images)
        
        response.success = True
        self.get_logger().info("All images has been processed and sent to inference succesfully.")
        
        return response

    def publish_batch_images(self, sample_of_images):
        for image in sample_of_images:
            image_path = self.dataset_dict[image]["img_path"]
            self.publish_image(image, image_path)
        
        while self._counter_images_sent_to_inference < self._batch_size:
            pass
        
        self._counter_images_sent_to_inference = 0
        self.get_logger().info(f"Inference counter reset to {self._counter_images_sent_to_inference}.")
        
    def publish_image(self, image_name, image_path):
        """Publish an image."""
        with open(image_path, "rb") as f:
            raw_data = f.read()

        width = 4656
        height = 3496
        msg = Image()
        msg.header.frame_id = image_name
        msg.height = height
        msg.width = width
        msg.encoding = "mono16"
        msg.step = width * 2
        msg.data = raw_data

        self.raw_image_publisher_.publish(msg)
        self.get_logger().info(f"RAW image with name {image_name}.raw published. Images sent to inference are {self._counter_images_sent_to_inference}")
        
    
    def callback_rgb_image(self, msg):
        """Callback function for processing RGB images."""
        self.get_logger().info("Received an RGB image.")

        image_name = msg.header.frame_id
        groundtruth_bounding_boxes = self.dataset_dict[image_name]["bboxes"]
        self.call_run_inference(msg, groundtruth_bounding_boxes)

    def call_run_inference(self, rgb_image, groundtruth_bounding_boxes):
        # Client topic to send rgb images to the ObjectDetector node
        client = self.create_client(RunInference, "run_inference")
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for Server run_inference from object_detector node...")
            
        request = RunInference.Request()
        request.image = rgb_image
        request.groundtruth_bounding_boxes = self.convert_list_to_boundingbox(groundtruth_bounding_boxes)
        
        future = client.call_async(request)
        future.add_done_callback(self.callback_call_run_inference)
        self.get_logger().info(f"Sent request to object_detector node.")
    
    def callback_call_run_inference(self, future):
        try:
            response = future.result()
            if response.success:
                self._counter_images_sent_to_inference += 1               
                self.get_logger().info(f"Image {self._counter_images_sent_to_inference} processed succesfully by the object_detector node.")
            else:               
                self.get_logger().info("Image failed while processing in object_detector node.")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))
            
    def convert_list_to_boundingbox(self, bounding_box_list):
        bounding_boxes_msg = []
        for bounding_box in bounding_box_list:
            bounding_box_msg = BoundingBox()
            bounding_box_msg.top_left_x = bounding_box[0]
            bounding_box_msg.top_left_y = bounding_box[1]
            bounding_box_msg.bottom_right_x = bounding_box[2]
            bounding_box_msg.bottom_right_y = bounding_box[3]
            bounding_box_msg.confidence_score = bounding_box[4]
            bounding_box_msg.class_name = bounding_box[5]
            bounding_boxes_msg.append(bounding_box_msg)
        return bounding_boxes_msg


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = DataLoaderNode()
    
    executor = MultiThreadedExecutor(4)
    
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
