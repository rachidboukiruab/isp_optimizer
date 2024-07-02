from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    classes_list_ = ["car", "person", "motorbike"]
    confidence_threshold = 0.5
    batch_size = 71
    verbose_level = 1
    isp_configuration_path = "./isp_configuration.json"
    annotations_folder_path = "/home/rachid/raw_dataset/train/"
    ld = LaunchDescription()
    
    isp_node = Node(
        package="hdr_isp",
        executable="hdr_isp",
        name="hdr_isp",
        arguments=[isp_configuration_path],
        parameters=[{"batch_size": batch_size},
                    {"verbose_level": verbose_level}]
        )

    data_loader_node = Node(
        package="isp_optimizer",
        executable="data_loader",
        name="data_loader",
        parameters=[{"images_folder_path": annotations_folder_path}, 
                    {"batch_size": batch_size},
                    {"verbose_level": verbose_level}]
    )
    
    object_detector_node = Node(
        package="isp_optimizer",
        executable="object_detector",
        name="object_detector",
        parameters=[{"confidence_threshold": confidence_threshold}, 
                    {"classes_list": classes_list_},
                    {"batch_size": batch_size},
                    {"show_image": False},
                    {"save_image": False},
                    {"output_folder": "./output_inference"},
                    {"verbose_level": verbose_level}]
    )
    
    cv_metrics_node = Node(
        package="isp_optimizer",
        executable="cv_metrics",
        name="cv_metrics",
        parameters=[{"annotations_folder_path": annotations_folder_path},
                    {"confidence_threshold": confidence_threshold}, 
                    {"classes_list": classes_list_},
                    {"batch_size": batch_size},
                    {"verbose_level": verbose_level}]
    )
    
    isp_optimizer_node = Node(
        package="isp_optimizer",
        executable="isp_optimizer",
        name="isp_optimizer",
        parameters=[{"isp_configuration_path": isp_configuration_path},
                    {"isp_tunner_path": "./isp_tuner_set1.json"},
                    {"json_results_path": "./results_01.json"},
                    {"verbose_level": verbose_level},
                    {"enable_space_reduction": True},
                    {"rounds_space_reduction": 1},
                    {"iterations_space_reduction": 100},
                    {"hyperparameters_normalization": True},
                    {"cma_initial_sigma": 0.5},
                    {"cma_population_size": 5},
                    {"cma_max_iterations": 200}]
    )

    ld.add_action(isp_node)
    ld.add_action(data_loader_node)
    ld.add_action(object_detector_node)
    ld.add_action(cv_metrics_node)
    #ld.add_action(isp_optimizer_node)
    return ld
