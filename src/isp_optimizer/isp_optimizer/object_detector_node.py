#!/usr/bin/env python3
# Standard library imports
from time import time_ns
# External package imports
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch

# Local package imports
from .utils import get_coco_name_from_id
from isp_optimizer_interfaces.msg import ListBoundingBox, BoundingBox

class ObjectDetectorNode(Node):
    """ Image detector and evaluation using YOLOv5n pretrained model"""
    def __init__(self):
        super().__init__("object_detector")

        # Parameters
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("classes_list", ["car"])
        self.declare_parameter("batch_size", 1)
        self.declare_parameter("show_image", True)
        self.declare_parameter("save_image", True)
        self.declare_parameter("output_folder", "./output_inference")
        #Verbose levels: 
        #    - 3 for debug logs
        #    - 2 for relevant user information
        #    - 1 for only warning and error logs
        #    - 0 for only error logs
        self.declare_parameter("verbose_level", 3)

        # Initialize parameters
        self._confidence_threshold = self.get_parameter("confidence_threshold").value
        self._classes_list = self.get_parameter("classes_list").value
        self._batch_size = self.get_parameter("batch_size").value
        self._show_image = self.get_parameter("show_image").value
        self._save_image = self.get_parameter("save_image").value
        self._output_folder = self.get_parameter("output_folder").value
        self._verbose_level = self.get_parameter("verbose_level").value
        self._counter_images_sent_to_evaluation = 0
        self.start_time = 0

        # Initialize bridge for converting ROS Image messages to OpenCV format
        self._ros_to_opencv_bridge = CvBridge()
        
        # Load and initialize Yolov5 model
        self._model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)
        # Define class colors
        self._class_colors = {
            "car": (128, 0, 128),  # Purple
            "person": (0, 0, 139),  # Dark red
            "motorbike": (255, 255, 0),  # Cyan
            "truck": (0, 139, 139)  # Yellow
        }
        
        # Subscriber where to receive processed images from ISP
        self.rgb_image_subscriber_ = self.create_subscription(
            Image, "rgb_image", self.callback_rgb_image, self._batch_size
        )
        
        # Publisher to send Predicted bounding boxes
        self.evaluation_publisher_ = self.create_publisher(
            ListBoundingBox, "run_evaluation", self._batch_size
        )

        # Log initialization
        if self._verbose_level >= 2:
            self.get_logger().info("Inference service from object_detector node initialized.")
            self.get_logger().info("Node parameters:")
            self.get_logger().info(f"   - confidence_threshold: {self._confidence_threshold}")
            self.get_logger().info(f"   - classes_list: {self._classes_list}")
            self.get_logger().info(f"   - show_image: {self._show_image}")
            self.get_logger().info(f"   - save_image: {self._save_image}")
            self.get_logger().info(f"   - output_folder: {self._output_folder}")
            
    def callback_rgb_image(self, msg):
        if self._counter_images_sent_to_evaluation == 0:
            self.start_time = time_ns()
        
        # Convert ROS Image message to OpenCV format
        if self._verbose_level >= 3:
            self.get_logger().info("Received an RGB image for evaluation.")
        try:
            opencv_image = self._ros_to_opencv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            if self._verbose_level >= 1:
                self.get_logger().warn("Error converting ROS Image to OpenCV format: %s" % str(e))      
        
        # process frame with yolov5n
        inference_result = self._model(opencv_image)
        predicted_bounding_boxes = self.convert_list_to_boundingbox(inference_result.xyxy[0])
        
        # Show and/or save image
        if self._show_image or self._save_image:
            bounding_boxes_image = self.draw_bounding_boxes(opencv_image, bounding_boxes = predicted_bounding_boxes, show_class_label = True)
            if self._show_image:
                self.show_image(bounding_boxes_image)
            if self._save_image:
                self.save_image(bounding_boxes_image, msg.header.frame_id)
        
        # Evaluate results
        evaluation_msg = ListBoundingBox()
        evaluation_msg.frame_id = msg.header.frame_id
        evaluation_msg.bounding_boxes = predicted_bounding_boxes
        self.evaluation_publisher_.publish(evaluation_msg)
        
        self._counter_images_sent_to_evaluation += 1
        if self._verbose_level >= 3:
            self.get_logger().info(f"Predictions from image {msg.header.frame_id} published. Remaining {self._batch_size - self._counter_images_sent_to_evaluation} images for inference.")

        if self._verbose_level >= 1 and self._counter_images_sent_to_evaluation >= self._batch_size:
            loop_time_ms = (time_ns() - self.start_time) / 1000000.
            self._counter_images_sent_to_evaluation = 0
            self.get_logger().info(f"All images has been sent to cv_metrics. Processing time: {loop_time_ms:.3f} ms.")

    def draw_bounding_boxes(self, image, bounding_boxes, show_class_label = False):
        for bounding_box in bounding_boxes:
            class_name = bounding_box.class_name
            if (class_name in self._classes_list) and (bounding_box.confidence_score >= self._confidence_threshold):
                top_left_x = bounding_box.top_left_x
                top_left_y = bounding_box.top_left_y
                bottom_right_x = bounding_box.bottom_right_x
                bottom_right_y = bounding_box.bottom_right_y                

                if show_class_label:
                    # bounding_box_label = f"{class_name}: {bounding_box[4]:.2f}"  # Class and confidence score
                    bounding_box_color = self._class_colors.get(class_name, (0, 0, 0))
                    cv2.putText(image, class_name, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, bounding_box_color, 5)
                else:
                    bounding_box_color = (0, 0, 0)
                
                # Draw bounding box in the image
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), bounding_box_color, 5)

        return image
    
    def save_image(self, image, image_name):
        os.makedirs(self._output_folder, exist_ok=True)            
        cv2.imwrite(self._output_folder+"/"+image_name+".png", image)
    
    def show_image(self, image):
        """
        :param image: Image array. Processed image.
        """
        # Show image with bounding boxes
        resized_image = self.resize_with_aspect_ratio(image, width=1280)  # Resize by width
        cv2.imshow("Detections", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """Resize an image while maintaining aspect ratio."""
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
    
    def convert_list_to_boundingbox(self, bounding_box_list):
        bounding_boxes_msg = []
        for bounding_box in bounding_box_list:
            bounding_box_msg = BoundingBox()
            bounding_box_msg.top_left_x = int(bounding_box[0])
            bounding_box_msg.top_left_y = int(bounding_box[1])
            bounding_box_msg.bottom_right_x = int(bounding_box[2])
            bounding_box_msg.bottom_right_y = int(bounding_box[3])
            bounding_box_msg.confidence_score = float(bounding_box[4])
            bounding_box_msg.class_name = get_coco_name_from_id(int(bounding_box[5]))
            bounding_boxes_msg.append(bounding_box_msg)
        return bounding_boxes_msg


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()