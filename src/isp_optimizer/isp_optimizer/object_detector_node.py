#!/usr/bin/env python3

# External package imports
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch

# Local package imports
from isp_optimizer_interfaces.srv import RunInference, RunEvaluation
from isp_optimizer_interfaces.msg import BoundingBox



class ObjectDetectorNode(Node):
    """ Image detector and evaluation using YOLOv5n pretrained model"""
    def __init__(self):
        super().__init__("object_detector")

        # Parameters
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("classes_list", ["car"])
        self.declare_parameter("show_image", True)
        self.declare_parameter("show_groundtruth", True)
        self.declare_parameter("save_image", True)
        self.declare_parameter("output_folder", "./output_inference")

        # Initialize parameters
        self._confidence_threshold = self.get_parameter("confidence_threshold").value
        self._classes_list = self.get_parameter("classes_list").value
        self._show_image = self.get_parameter("show_image").value
        self._show_groundtruth = self.get_parameter("show_groundtruth").value
        self._save_image = self.get_parameter("save_image").value
        self._output_folder = self.get_parameter("output_folder").value

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
        
        # Server Initializer
        self._server_ = self.create_service(
            RunInference, "run_inference", self.callback_run_inference)

        # Log initialization
        self.get_logger().info("Inference service from object_detector node initialized.")
        self.get_logger().info("Node parameters:")
        self.get_logger().info(f"   - confidence_threshold: {self._confidence_threshold}")
        self.get_logger().info(f"   - classes_list: {self._classes_list}")
        self.get_logger().info(f"   - show_image: {self._show_image}")
        self.get_logger().info(f"   - show_groundtruth: {self._show_groundtruth}")
        self.get_logger().info(f"   - save_image: {self._save_image}")
        self.get_logger().info(f"   - output_folder: {self._output_folder}")
    
    def callback_run_inference(self, request, response):
        # Convert ROS Image message to OpenCV format
        self.get_logger().info("Received an RGB image for evaluation.")
        try:
            opencv_image = self._ros_to_opencv_bridge.imgmsg_to_cv2(request.image, "bgr8")
        except CvBridgeError as e:
            response.success = False
            self.get_logger().info("Error converting ROS Image to OpenCV format: %s" % str(e))      
            return response
        
        # process frame with yolov5n
        inference_result = self._model(opencv_image)
        predicted_bounding_boxes = self.convert_list_to_boundingbox(inference_result.xyxy[0])
        groundtruth_bounding_boxes = request.groundtruth_bounding_boxes
        
        # Show and/or save image
        if self._show_image or self._save_image:
            bounding_boxes_image = self.draw_bounding_boxes(opencv_image, bounding_boxes = predicted_bounding_boxes, show_class_label = True)
            if self._show_groundtruth:
                bounding_boxes_image = self.draw_bounding_boxes(bounding_boxes_image, bounding_boxes = groundtruth_bounding_boxes)
            if self._show_image:
                self.show_image(bounding_boxes_image)
            if self._save_image:
                self.save_image(bounding_boxes_image, request.image.header.frame_id)
        
        # Evaluate results
        self.call_evaluate_metrics(predicted_bounding_boxes, groundtruth_bounding_boxes)
        
        response.success = True
        self.get_logger().info("Inference and evaluation done successfully.")
        
        return response
    
    def call_evaluate_metrics(self, predicted_bounding_boxes, groundtruth_bounding_boxes):
        # Client topic to send bounding boxes to the cv_metrics node
        client = self.create_client(RunEvaluation, "run_evaluation")
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for Server run_inference from object_detector node...")
            
        request = RunEvaluation.Request()
        request.predicted_bounding_boxes = predicted_bounding_boxes
        request.groundtruth_bounding_boxes = groundtruth_bounding_boxes
        
        future = client.call_async(request)
        future.add_done_callback(self.callback_call_evaluate_metrics)
    
    def callback_call_evaluate_metrics(self, future):
        try:
            response = future.result()
            if response.success:            
                self.get_logger().info(f"Evaluation done successfully.")
            else:               
                self.get_logger().info("Evaluation failed.")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))

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
            bounding_box_msg.class_name = self.get_coco_name_from_id(int(bounding_box[5]))
            bounding_boxes_msg.append(bounding_box_msg)
        return bounding_boxes_msg
          
    def get_coco_name_from_id(self, class_id):
        id_to_name = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorbike',
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

        return id_to_name[class_id]


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()