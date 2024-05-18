#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class CMAESOptimizerNode(Node):
    def __init__(self):
        super().__init__("cma_es_optimizer")
        self.call_process_batch_images()
    
    def call_process_batch_images(self):
        # Client topic to send rgb images to the ObjectDetector node
        client = self.create_client(Trigger, "process_batch_images")
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for Server process_batch_images from data_loader node...")
            
        request = Trigger.Request()
        
        future = client.call_async(request)
        future.add_done_callback(self.callback_call_process_batch_images)
    
    def callback_call_process_batch_images(self, future):
        try:
            response = future.result()
            if response.success:              
                self.get_logger().info(f"Trigger sent succesfully to the data_loader node.")
            else:               
                self.get_logger().info("Failed while processing in data_loader node.")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))


def main(args=None):
    rclpy.init(args=args)
    node = CMAESOptimizerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
