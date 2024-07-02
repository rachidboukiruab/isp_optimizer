from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    verbose_level = 3
    isp_configuration_path = "./isp_configuration.json"
    ld = LaunchDescription()
    
    isp_optimizer_node = Node(
        package="isp_optimizer",
        executable="isp_optimizer",
        name="isp_optimizer",
        parameters=[{"isp_configuration_path": isp_configuration_path},
                    {"isp_tunner_path": "./isp_tuner_set2.json"},
                    {"json_results_path": "./results_07.json"},
                    {"verbose_level": verbose_level},
                    {"enable_space_reduction": False},
                    {"rounds_space_reduction": 1},
                    {"iterations_space_reduction": 100},
                    {"hyperparameters_normalization": True},
                    {"cma_initial_sigma": 0.5},
                    {"cma_population_size": 5},
                    {"cma_max_iterations": 200}]
    )

    ld.add_action(isp_optimizer_node)
    return ld
