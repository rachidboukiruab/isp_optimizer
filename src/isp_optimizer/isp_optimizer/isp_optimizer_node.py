#!/usr/bin/env python3

# Standard library imports
import os
import json
from math import log10
from time import sleep, time_ns

# External package imports
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
import cma

# Local package imports
from isp_optimizer_interfaces.srv import GetMetricResults
from isp_optimizer_interfaces.msg import ComputerVisionMetrics


class IspOptimizerNode(Node):
    def __init__(self):
        super().__init__("isp_optimizer")
        
        # Parameters
        self.declare_parameter("isp_configuration_path", "./isp_configuration.json")
        self.declare_parameter("isp_tunner_path", "./isp_tuner.json")        
        self.declare_parameter("verbose_level", 3)
        """
        Verbose levels: 
            - 3 for debug logs
            - 2 for relevant user information
            - 1 for only warning and error logs
            - 0 for only error logs
        """ 
        
        # Initialize parameters
        self._isp_configuration_path = self.get_parameter("isp_configuration_path").value
        self._isp_tunner_path = self.get_parameter("isp_tunner_path").value
        self._verbose_level = self.get_parameter("verbose_level").value
        
        self._isp_configuration_dict = self.read_json_as_dict(self._isp_configuration_path)
        self._isp_tunner_dict = self.read_json_as_dict(self._isp_tunner_path)
        
        self._optimization_loops_counter = 0

        # Publisher initializer
        self.isp_load_new_configuration_ = self.create_publisher(
            String, "isp_json_path", 10
        )
        
        # Log initialization
        if self._verbose_level >= 2:
            self.get_logger().info("ISP Optimizer node initialized.")
        
        self.optimize_parameters_using_cma()
        if self._verbose_level >= 2:
            self.get_logger().info("ISP optimization finished.")
        
    def publish_isp_configuration_path(self):
        msg = String()
        msg.data = self._isp_configuration_path
        
        self.isp_load_new_configuration_.publish(msg)
    
    def call_process_batch_images(self):
        # Client topic to send rgb images to the ObjectDetector node
        client = self.create_client(Trigger, "process_batch_images")
        while not client.wait_for_service(1.0):
            if self._verbose_level >= 1:
                self.get_logger().warn("Waiting for Server process_batch_images from data_loader node...")
            
        request = Trigger.Request()
        
        future = client.call_async(request)
        future.add_done_callback(self.callback_call_process_batch_images)
    
    def callback_call_process_batch_images(self, future):
        try:
            response = future.result()
            if response.success:
                if self._verbose_level >= 3:              
                    self.get_logger().info(f"Trigger sent succesfully to the data_loader node.")
            else:
                if self._verbose_level >= 1:               
                    self.get_logger().warn("Failed while processing in data_loader node.")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))
            
    def get_cv_metric_results(self):
        # Client topic to receive mAP and mAR from cv_metrics node
        client = self.create_client(GetMetricResults, "cv_metric_results")
        while not client.wait_for_service(1.0):
            if self._verbose_level >= 1:
                self.get_logger().warn("Waiting for Server cv_metrics_result from cv_metrics node...")
            
        request = GetMetricResults.Request()
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def get_map_and_mar(self):
        response = self.get_cv_metric_results()
        while not response.success:
            sleep(0.1)
            response = self.get_cv_metric_results()
        
        result_msg = response.result
        mean_average_precision = result_msg.mean_average_precision
        mean_average_recall = result_msg.mean_average_recall
        return mean_average_precision, mean_average_recall
        
    
    def get_cost_function_loop_result(self, x):
        """
        Definition of the cost function that will be used in the optimizers. This function loads the 
        proposed X values for the ISP hyperparameters, processes the batch of the training images with the
        new proposed hyperparameters, performs the inference and extracts the mAP of the detections and returns 
        1-(mAP*0.5 + mAR*0.5) as the loss we want to minimize.
        x: list of the proposed values for the ISP hyperparameters
        :return: The loss function that we want to minimize (1 - mAP)
        """
        start_time = time_ns()
        text = ""
        i = 0
        # Load the hyperparameters proposed by the optimizer in the ISP config file
        for module in self._isp_tunner_dict:
            for param in self._isp_tunner_dict[module]:
                if param == "gammalut":
                    self.set_rgb_gamma_lut(x[i])
                else:
                    self._isp_configuration_dict[module][param] = x[i]
                text += '{}/{}: {:.2f} \t'.format(module, param, x[i])
                i += 1
        
        # Save new parameters in a JSON that will be used by the ISP
        self.save_dict_as_json(self._isp_configuration_dict, self._isp_configuration_path)
        # Send a flag to the ISP to know that we have a new ISP parameters proposal and has to reload them
        self.publish_isp_configuration_path()
        
        # Send a flag to the dataloader to start feeding the ISP with raw images
        self.call_process_batch_images()
        
        # Wait to receive the computed mAP and mAR from cv_metrics_node
        mean_average_precision, mean_average_recall = self.get_map_and_mar()
        
        # Measure computation time in milliseconds and print results
        loop_time_ms = (time_ns() - start_time) / 1000000.
        self._optimization_loops_counter += 1
        if self._verbose_level >= 2:
            self.get_logger().info(f'{self._optimization_loops_counter}: \t{text} mAP: {mean_average_precision*100:.3f}% \tmAR: {mean_average_recall*100:.3f}% \tComputation Time (ms): {loop_time_ms:.3f}')
        
        # Calculate cost function result
        cost_function = 1 - (mean_average_precision*0.5 + mean_average_recall*0.5)
        return cost_function
    
    def optimize_parameters_using_cma(self):
        """
        Optimizes the ISP hyperparameters using Covariance Matrix Adaptation (CMA) algorithm from cma.fmin2 object.
        :return: Tuple of a list of the ISP hyperparameters that minimizes the defined cost function and a dictionary
                of the detail of hyperparameters used in each iteration and the result in IoU, mAP, Precision, Recall.
        """
        initial_values = [self._isp_tunner_dict[module][param]["initial_value"] for module in self._isp_tunner_dict for param in self._isp_tunner_dict[module]]
        lower_bounds = [self._isp_tunner_dict[module][param]["lower_bound"] for module in self._isp_tunner_dict for param in self._isp_tunner_dict[module]]
        upper_bounds = [self._isp_tunner_dict[module][param]["upper_bound"] for module in self._isp_tunner_dict for param in self._isp_tunner_dict[module]]

        xopt, es = cma.fmin2(self.get_cost_function_loop_result, initial_values, 1, {'bounds': [lower_bounds, upper_bounds]})
        return xopt

        
    def generate_gamma_curve(self, gamma_value, number_points = 11):
        gamma_curve = []
        input_slice_value = 1.0 / (number_points - 1)
        
        for i in range(number_points):
            if i == 0:
                gamma_curve.append(0.0)
            else:
                input_value = input_slice_value * i
                output_value = 10**(1/gamma_value*log10(input_value))
                gamma_curve.append(output_value)
        
        return gamma_curve
    
    def set_rgb_gamma_lut(self, gamma_value):
        new_gamma_lut = self.generate_gamma_curve(gamma_value, self._isp_configuration_dict["rgbgamma"]["gammalut_nums"])
        self._isp_configuration_dict["rgbgamma"]["gammalut"] = new_gamma_lut
    
    def normalize_isp_parameter(self, isp_parameter, lowest_value, highest_value):
        normalized_isp_parameter = (isp_parameter - lowest_value) / float(highest_value - lowest_value)
        return normalized_isp_parameter
    
    def denormalize_isp_parameter(self, normalized_isp_parameter, lowest_value, highest_value):
        isp_parameter = normalized_isp_parameter * (highest_value - lowest_value) + lowest_value
        return isp_parameter
    
    def read_json_as_dict(self, file_path):
        """
        Reads a JSON file and returns its contents as a dictionary.
        
        Parameters:
        file_path (str): The path to the JSON file.

        Returns:
        dict: The contents of the JSON file as a dictionary.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def save_dict_as_json(self, data_dict, file_path):
        """
        Saves a dictionary as a JSON file. Creates any necessary directories.
        
        Parameters:
        data_dict (dict): The dictionary to save as JSON.
        file_path (str): The path to the JSON file to be created.
        """
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=4)


def main(args=None):
    rclpy.init(args=args)
    node = IspOptimizerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
