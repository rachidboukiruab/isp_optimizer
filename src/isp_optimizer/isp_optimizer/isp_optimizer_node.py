#!/usr/bin/env python3

# Standard library imports
import os
import json
from math import log10
from time import sleep, time_ns
import copy

# External package imports
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
import cma
from skopt import dummy_minimize
from skopt.space import Real
import numpy as np

# Local package imports
from isp_optimizer_interfaces.srv import GetMetricResults


class IspOptimizerNode(Node):
    def __init__(self):
        super().__init__("isp_optimizer")
        
        # Declare parameters
        self.declare_parameter("isp_configuration_path", "./isp_configuration.json")
        self.declare_parameter("isp_tunner_path", "./isp_tuner.json")    
        self.declare_parameter("json_results_path", "./results.json")    
        self.declare_parameter("verbose_level", 3)
        self.declare_parameter("enable_space_reduction", False)
        self.declare_parameter("rounds_space_reduction", 1)
        self.declare_parameter("iterations_space_reduction", 100)
        self.declare_parameter("hyperparameters_normalization", True)
        self.declare_parameter("cma_initial_sigma", 0.5)
        self.declare_parameter("cma_evaluate_initial_x", True)
        self.declare_parameter("cma_population_size", 5)
        self.declare_parameter("cma_max_iterations", 200)
        self.declare_parameter("cma_tolerance_x", 1e-6)
        self.declare_parameter("cma_tolerance_loss", 1e-5)
        self.declare_parameter("cma_restarts", 0)
        self.declare_parameter("cma_restart_from_best", False)
        self.declare_parameter("cma_multiplier_increase_population_size", 2)
        self.declare_parameter("cma_bipopulation", False)
        
        # Initialize parameters
        self._isp_configuration_path = self.get_parameter("isp_configuration_path").value
        self._isp_tunner_path = self.get_parameter("isp_tunner_path").value
        self._json_results_path = self.get_parameter("json_results_path").value
        self._verbose_level = self.get_parameter("verbose_level").value
        self._enable_space_reduction = self.get_parameter("enable_space_reduction").value
        self._rounds_space_reduction = self.get_parameter("rounds_space_reduction").value
        self._iterations_space_reduction = self.get_parameter("iterations_space_reduction").value
        self._hyperparameters_normalization = self.get_parameter("hyperparameters_normalization").value
        self._cma_initial_sigma = self.get_parameter("cma_initial_sigma").value
        self._cma_evaluate_initial_x = self.get_parameter("cma_evaluate_initial_x").value
        self._cma_population_size = self.get_parameter("cma_population_size").value
        self._cma_max_iterations = self.get_parameter("cma_max_iterations").value
        self._cma_tolerance_x = self.get_parameter("cma_tolerance_x").value
        self._cma_tolerance_loss = self.get_parameter("cma_tolerance_loss").value
        self._cma_restarts = self.get_parameter("cma_restarts").value
        self._cma_restart_from_best = self.get_parameter("cma_restart_from_best").value
        self._cma_multiplier_increase_population_size = self.get_parameter("cma_multiplier_increase_population_size").value
        self._cma_bipopulation = self.get_parameter("cma_bipopulation").value

        # Read configurations from JSON files
        self._isp_configuration_dict = self.read_json_as_dict(self._isp_configuration_path)
        self._isp_tunner_dict = self.read_json_as_dict(self._isp_tunner_path)
        
        # Initialize results dictionary and bounds lists
        self._metrics_dictionary = self.initialize_results_dict()        
        self._lower_bounds, self._upper_bounds = self.get_bounds()       
        self._optimization_loops_counter = 0
        self._normalization_applied = False

        # Initialize publisher for ISP configuration path
        self.isp_load_new_configuration_ = self.create_publisher(
            String, "isp_json_path", 10
        )
        
        # Log initialization
        if self._verbose_level >= 2:
            self.get_logger().info("ISP Optimizer node initialized.")
        
        # Start optimization process
        self.optimize_parameters_using_cma()
        if self._verbose_level >= 2:
            self.get_logger().info("ISP optimization finished.")
        
    def publish_isp_configuration_path(self):
        """
        Publish the ISP configuration path to trigger the loading of new parameters.

        Args:
            None

        Returns:
            None
        """
        msg = String()
        msg.data = self._isp_configuration_path
        
        self.isp_load_new_configuration_.publish(msg)
    
    def call_process_batch_images(self):
        """
        Call the process_batch_images service to process a batch of images.

        Args:
            None

        Returns:
            None
        """
        # Client topic to send rgb images to the ObjectDetector node
        client = self.create_client(Trigger, "process_batch_images")
        while not client.wait_for_service(1.0):
            if self._verbose_level >= 1:
                self.get_logger().warn("Waiting for Server process_batch_images from data_loader node...")
            
        request = Trigger.Request()
        
        future = client.call_async(request)
        future.add_done_callback(self.callback_call_process_batch_images)
    
    def callback_call_process_batch_images(self, future):
        """
        Callback function for processing the response of process_batch_images.

        Args:
            future (Future): The future object containing the service call result.

        Returns:
            None
        """
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
        """
        Call the cv_metric_results service to get the mAP and mAR results.

        Args:
            None

        Returns:
            GetMetricResults.Response: The response object containing the metric results.
        """
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
        """
        Retrieve mean average precision (mAP) and mean average recall (mAR) from the metrics service.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - mean_average_precision (float): The mAP value.
                - mean_average_recall (float): The mAR value.
        """
        response = self.get_cv_metric_results()
        while not response.success:
            sleep(0.05)
            response = self.get_cv_metric_results()
        
        result_msg = response.result
        mean_average_precision = result_msg.mean_average_precision
        mean_average_recall = result_msg.mean_average_recall
        return mean_average_precision, mean_average_recall
        
    
    def get_cost_function_loop_result(self, x):
        """
        Calculate the cost function for the optimization process.

        Args:
            x (list): List of proposed values for the ISP hyperparameters.

        Returns:
            float: The loss value that needs to be minimized.
        """
        start_time = time_ns()
        text = ""
        i = 0
                
        # Load the hyperparameters proposed by the optimizer in the ISP config file
        for module in self._isp_tunner_dict:
            if module == "blc":
                hyperparameter = self.denormalize_isp_parameter(x[i], self._lower_bounds[i], self._upper_bounds[i]) if self._normalization_applied else int(x[i])
                self._isp_configuration_dict[module] = hyperparameter
                i += 1
                text += '{}: {:.2f} \t'.format(module, hyperparameter)
                self._metrics_dictionary['Hyperparameters'][module][module].append(hyperparameter)
            else:
                for param in self._isp_tunner_dict[module]:
                    if param == "gammalut":
                        if isinstance(self._isp_tunner_dict[module][param]['initial_value'], list):
                            gammalut_list = []
                            for index in range(len(self._isp_tunner_dict[module][param]['initial_value'])):
                                hyperparameter = self.denormalize_isp_parameter(x[i], self._lower_bounds[i], self._upper_bounds[i]) if self._normalization_applied else x[i]
                                gammalut_list.append(hyperparameter)
                                i +=1       
                            self._isp_configuration_dict[module][param] = gammalut_list
                            self._metrics_dictionary['Hyperparameters'][module][param].append(gammalut_list)
                            # Format the gammalut_list to a string with each element formatted to two decimal places
                            gammalut_str = ', '.join(f'{value:.2f}' for value in gammalut_list)
                            text += '{}/{}: [{}] \t'.format(module, param, gammalut_str)
                        else:
                            hyperparameter = self.denormalize_isp_parameter(x[i], self._lower_bounds[i], self._upper_bounds[i]) if self._normalization_applied else x[i]
                            new_gamma_lut = self.generate_gamma_curve(hyperparameter, self._isp_configuration_dict[module]["gammalut_nums"])
                            self._isp_configuration_dict[module][param] = new_gamma_lut
                            i += 1
                            text += '{}/{}: {:.2f} \t'.format(module, param, hyperparameter)
                            self._metrics_dictionary['Hyperparameters'][module][param].append(hyperparameter)
                    elif module == "wb_gain":
                        wb_list = []
                        for index in range(len(self._isp_tunner_dict[module][param]['initial_value'])):
                            hyperparameter = self.denormalize_isp_parameter(x[i], self._lower_bounds[i], self._upper_bounds[i]) if self._normalization_applied else float(x[i])
                            wb_list.append(hyperparameter)
                            i +=1  
                        self._isp_configuration_dict[module][param] = wb_list
                        self._metrics_dictionary['Hyperparameters'][module][param].append(wb_list)
                        # Format the gammalut_list to a string with each element formatted to two decimal places
                        wb_str = ', '.join(f'{value:.2f}' for value in wb_list)
                        text += '{}/{}: [{}] \t'.format(module, param, wb_str)                        
                    else:
                        hyperparameter = self.denormalize_isp_parameter(x[i], self._lower_bounds[i], self._upper_bounds[i]) if self._normalization_applied else int(x[i])
                        self._isp_configuration_dict[module][param] = hyperparameter
                        i += 1
                        text += '{}/{}: {:.2f} \t'.format(module, param, hyperparameter)
                        self._metrics_dictionary['Hyperparameters'][module][param].append(hyperparameter)
                      
        # Save new parameters in a JSON that will be used by the ISP
        self.save_dict_as_json(self._isp_configuration_dict, self._isp_configuration_path)
        
        # Publish the new ISP configuration path
        self.publish_isp_configuration_path()
        
        # Trigger the process to start feeding the ISP with raw images
        self.call_process_batch_images()
        
        # Wait to receive the computed mAP and mAR from cv_metrics_node
        mean_average_precision, mean_average_recall = self.get_map_and_mar()
        
        # Update metrics dictionary
        self._metrics_dictionary['mAP'].append(mean_average_precision)
        self._metrics_dictionary['mAR'].append(mean_average_recall)
        
        # Measure computation time in milliseconds and print results
        loop_time_ms = (time_ns() - start_time) / 1000000.
        self._optimization_loops_counter += 1
        if self._verbose_level >= 2:
            self.get_logger().info(f'{self._optimization_loops_counter}: \t{text} mAP: {mean_average_precision*100:.3f}% \tmAR: {mean_average_recall*100:.3f}% \tComputation Time (ms): {loop_time_ms:.3f}')
        
        # Calculate cost function result
        cost_function = 1 - (mean_average_precision*1.0 + mean_average_recall*0.0)
        return cost_function
    
    def optimize_parameters_using_cma(self):
        """
        Optimizes the ISP hyperparameters using Covariance Matrix Adaptation (CMA) algorithm from cma.fmin2 object.
        
        Returns:
            tuple: A tuple containing:
                - A list of the ISP hyperparameters that minimizes the defined cost function
                - A dictionary of the detail of hyperparameters used in each iteration and the result in IoU, mAP, Precision, Recall.
        """
        
        if self._enable_space_reduction:
            self.hyperparameters_space_reduction(number_rounds=self._rounds_space_reduction, iterations_per_round=self._iterations_space_reduction)
            sorted_hyperparameters = self.sort_hyperparameters()
            initial_values = self.get_initial_values(sorted_hyperparameters)
            
            if self._verbose_level >= 2:
                self.get_logger().info(f'Space Reduction done. Best solution obtained is: {initial_values}')
        else:            
            initial_values = self.get_initial_values()
        
        if self._hyperparameters_normalization:
            initial_values = [self.normalize_isp_parameter(initial_values[i], self._lower_bounds[i], self._upper_bounds[i]) for i in range(len(initial_values))]
            lower_bounds = [self.normalize_isp_parameter(self._lower_bounds[i], self._lower_bounds[i], self._upper_bounds[i]) for i in range(len(self._lower_bounds))]
            upper_bounds = [self.normalize_isp_parameter(self._upper_bounds[i], self._lower_bounds[i], self._upper_bounds[i]) for i in range(len(self._upper_bounds))]
        else:
            lower_bounds = self._lower_bounds
            upper_bounds = self._upper_bounds

        try:
            self._normalization_applied = True if self._hyperparameters_normalization else False
            # Run the optimizer
            xopt, es = cma.fmin2(
                self.get_cost_function_loop_result,                         # The cost function defined above
                x0=initial_values,                                          # Initial guesses for hyperparameters
                sigma0=self._cma_initial_sigma,                             # Initial standard deviation for parameter updates
                eval_initial_x=self._cma_evaluate_initial_x,                # Evaluate initial set of hyperparameters
                options={
                    'bounds': [lower_bounds, upper_bounds],                 # Bounds as a list of lists
                    'popsize': self._cma_population_size,                   # Population size
                    'maxiter': self._cma_max_iterations,                    # Maximum iterations
                    'tolx': self._cma_tolerance_x,                          # Tolerance for parameter change
                    'tolfun': self._cma_tolerance_loss                      # Tolerance for function value change
                },
                restarts= self._cma_restarts,                               # Number of restarts
                restart_from_best= self._cma_restart_from_best,             # Restart from best solution
                incpopsize= self._cma_multiplier_increase_population_size,  # Increase population size with each restart
                bipop= self._cma_bipopulation,                              # Use bipop strategy for robustness
            )

            self.save_dict_as_json(self._metrics_dictionary, self._json_results_path)
            
            # Output results
            if self._verbose_level >= 2:
                self.get_logger().info(f'Optimal hyperparameters: {xopt}')
                self.get_logger().info(f'Optimization result: {es.result}')
            return xopt
        except KeyboardInterrupt:
            self.save_dict_as_json(self._metrics_dictionary, self._json_results_path)
            return None       

    def hyperparameters_space_reduction(self, number_rounds, iterations_per_round):
        """
        Reduce the boundaries of hyperparameters.
        
        Args:
            number_rounds (int): Number of reduction rounds.
            iterations_per_round (int): Number of iterations per round.
        """
        self._normalization_applied = False       
        for i in range(number_rounds):
            # Set Bounds
            bounds = list(zip(self._lower_bounds, self._upper_bounds))
            dummy_minimize(self.get_cost_function_loop_result, bounds, n_calls=iterations_per_round, random_state=0)
            
            # Sort hyperparameters based on mAP
            sorted_hyperparameters = self.sort_hyperparameters()
            
            # Reduce the bounds according to the best results obtained
            self._lower_bounds, self._upper_bounds = self.reduce_bounds(sorted_hyperparameters)
            if self._verbose_level >= 3:
                self.get_logger().info(f'Space Reduction iteration {i} done.\n' +
                                       f'- New lower bounds are: {self._lower_bounds}\n' +
                                       f'- New upper bounds are: {self._upper_bounds}')
                
    def sort_hyperparameters(self):
        """
        Sort hyperparameters based on mean Average Precision (mAP).
        
        Returns:
            dict: Sorted hyperparameters.
        """
        # get the indices that would sort the array in ascending order
        ascending_indices = np.argsort(self._metrics_dictionary["mAP"])

        # reverse the ascending indices to get descending indices
        descending_indices = ascending_indices[::-1]

        hyperparameters = copy.deepcopy(self._metrics_dictionary["Hyperparameters"])
        sorted_hyperparameters_descending = copy.deepcopy(self._metrics_dictionary["Hyperparameters"])
        for module in hyperparameters:
            for param in hyperparameters[module]:
                    hyperparameters[module][param] = np.array(hyperparameters[module][param])
                    sorted_hyperparameters_descending[module][param] = hyperparameters[module][param][descending_indices]
        
        return sorted_hyperparameters_descending
    
    def reduce_bounds(self, sorted_hyperparameters):
        """
        Reduce the bounds of hyperparameters based on the top-performing hyperparameters.
        
        Args:
            sorted_hyperparameters (dict): Sorted hyperparameters.
        
        Returns:
            tuple: A tuple containing:
                - A list of new lower bounds
                - A list of new upper bounds
        """
        
        top_fraction = 0.5  # Use top 50% of the hyperparameters to reduce bounds
        num_top = int(len(self._metrics_dictionary["mAP"]) * top_fraction)
        
        new_lower_bounds = []
        new_upper_bounds = []
        
        for module in sorted_hyperparameters:
            for param in sorted_hyperparameters[module]:
                top_values = sorted_hyperparameters[module][param][:num_top]
                test = top_values[0]
                if isinstance(test, np.ndarray):
                    min_values = np.min(top_values, axis = 0)
                    new_lower_bounds.extend(min_values.tolist())
                    max_values = np.max(top_values, axis = 0)
                    new_upper_bounds.extend(max_values.tolist())
                else:
                    new_lower_bounds.append(int(np.min(top_values)))
                    new_upper_bounds.append(int(np.max(top_values)))
        
        return new_lower_bounds, new_upper_bounds

        
    def get_initial_values(self, hyperparameters_dict = None):
        """
        Get initial values for hyperparameters.
        
        Args:
            hyperparameters_dict (dict): Dictionary containing initial hyperparameters (optional).
            
        Returns:
            list: List of initial hyperparameter values.
        """
        initial_values = []
        for module in self._isp_tunner_dict:
            if module == "blc":
                value = self._isp_tunner_dict[module]["initial_value"] if hyperparameters_dict is None else hyperparameters_dict[module][module][0]
                if isinstance(value, list):
                    initial_values.extend(value)
                elif isinstance(value, np.ndarray):
                    initial_values.extend(value.tolist())
                elif isinstance(value, np.generic):
                    initial_values.append(int(value))
                else:
                    initial_values.append(value)
            else:
                for param in self._isp_tunner_dict[module]:
                    value = self._isp_tunner_dict[module][param]["initial_value"] if hyperparameters_dict is None else hyperparameters_dict[module][param][0]                        
                    # Check if the value is a list (specifically for "gammalut" parameters)
                    if isinstance(value, list):
                        initial_values.extend(value)
                    elif isinstance(value, np.ndarray):
                        initial_values.extend(value.tolist())
                    elif isinstance(value, np.generic):
                        initial_values.append(int(value))
                    else:
                        initial_values.append(value)
        return initial_values
    
    def get_bounds(self):
        """
        Get lower and upper bounds for hyperparameters.
        
        Returns:
            tuple: A tuple containing:
                - A list of lower bounds for hyperparameters
                - A list of upper bounds for hyperparameters
        """
        lower_bounds = []
        upper_bounds = []
        for module in self._isp_tunner_dict:
            if module == "blc":
                lower_bound = self._isp_tunner_dict[module]["lower_bound"]
                upper_bound = self._isp_tunner_dict[module]["upper_bound"]
                # Append the single lower and upper bounds values
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
            else:
                for param in self._isp_tunner_dict[module]:
                    initial_value = self._isp_tunner_dict[module][param]["initial_value"]
                    lower_bound = self._isp_tunner_dict[module][param]["lower_bound"]
                    upper_bound = self._isp_tunner_dict[module][param]["upper_bound"]
                    
                    # Check if the initial value is a list (specifically for "gammalut" parameters)
                    if isinstance(initial_value, list) and isinstance(lower_bound, list):
                        if len(initial_value) == len(lower_bound) and len(initial_value) == len(upper_bound):
                            # Extend the lower and upper bounds to match the length of the initial values list
                            lower_bounds.extend(lower_bound)
                            upper_bounds.extend(upper_bound)
                        else:
                            self.get_logger().error(f'Size of initial_value, lower_bound, and upper_bound are not equal.')
                            num_values = len(initial_value)
                            lower_bounds.extend([lower_bound[0]] * num_values)
                            upper_bounds.extend([upper_bound[0]] * num_values)
                    elif isinstance(initial_value, list):
                        # Extend the lower and upper bounds to match the length of the initial values list
                        num_values = len(initial_value)
                        lower_bounds.extend([lower_bound] * num_values)
                        upper_bounds.extend([upper_bound] * num_values)
                    else:
                        # Append the single lower and upper bounds values
                        lower_bounds.append(lower_bound)
                        upper_bounds.append(upper_bound)
        return lower_bounds, upper_bounds
            
    def generate_gamma_curve(self, gamma_value, number_points = 11):
        """
        Generate a gamma curve.
        
        Args:
            gamma_value (float): The gamma value.
            number_points (int): Number of points on the gamma curve (default is 11).
        
        Returns:
            list: List of gamma curve values.
        """
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
    
    def normalize_isp_parameter(self, isp_parameter, lowest_value, highest_value):
        """
        Normalize ISP parameter.
        
        Args:
            isp_parameter (float): The ISP parameter value.
            lowest_value (float): The lowest possible value for the parameter.
            highest_value (float): The highest possible value for the parameter.
        
        Returns:
            float: The normalized ISP parameter value.
        """
        normalized_isp_parameter = 1 if highest_value == lowest_value else (isp_parameter - lowest_value) / float(highest_value - lowest_value)
        return normalized_isp_parameter
    
    def denormalize_isp_parameter(self, normalized_isp_parameter, lowest_value, highest_value):
        """
        Denormalize ISP parameter.
        
        Args:
            normalized_isp_parameter (float): The normalized ISP parameter value.
            lowest_value (float): The lowest possible value for the parameter.
            highest_value (float): The highest possible value for the parameter.
        
        Returns:
            float: The denormalized ISP parameter value.
        """
        isp_parameter = highest_value if highest_value == lowest_value else normalized_isp_parameter * (highest_value - lowest_value) + lowest_value
        return isp_parameter
    
    def read_json_as_dict(self, file_path):
        """
        Reads a JSON file and returns its contents as a dictionary.
        
        Args:
            file_path (str): The path to the JSON file.
        
        Returns:
            dict: The contents of the JSON file as a dictionary.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def save_dict_as_json(self, data_dict, file_path):
        """"
        Saves a dictionary as a JSON file. Creates any necessary directories.
        
        Args:
            data_dict (dict): The dictionary to save as JSON.
            file_path (str): The path to the JSON file to be created.
        """
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=4)
    
    def initialize_results_dict(self):
        """
        Initialize a dictionary for storing metrics results.
        
        Returns:
            dict: The initialized metrics dictionary.
        """
        mean_metrics_dict = {'Hyperparameters': {}} 
        for module in self._isp_tunner_dict:
            mean_metrics_dict['Hyperparameters'][module] = {}
            if module == 'blc':
                mean_metrics_dict['Hyperparameters'][module][module] = []
            else:
                for param in self._isp_tunner_dict[module]:
                    mean_metrics_dict['Hyperparameters'][module][param] = []
                

        mean_metrics_dict['mAP'] = []
        mean_metrics_dict['mAR'] = []
        return mean_metrics_dict


def main(args=None):
    """Main function to initialize and spin the IspOptimizerNode."""
    rclpy.init(args=args)
    node = IspOptimizerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
