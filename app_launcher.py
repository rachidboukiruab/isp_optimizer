from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    classes_list_ = ["car"]
    confidence_threshold = 0.5
    ld = LaunchDescription()
    
    isp_node = Node(
        package="hdr_isp",
        executable="hdr_isp",
        name="hdr_isp",
        arguments=["./isp_configuration.json"]
        )

    data_loader_node = Node(
        package="isp_optimizer",
        executable="data_loader",
        name="data_loader",
        parameters=[{"annotations_folder_path": "/home/rachid/raw_dataset/test/"}, 
                    {"classes_list": classes_list_},
                    {"batch_size": 10}]
    )
    
    object_detector_node = Node(
        package="isp_optimizer",
        executable="object_detector",
        name="object_detector",
        parameters=[{"confidence_threshold": confidence_threshold}, 
                    {"classes_list": classes_list_},
                    {"show_image": False},
                    {"show_groundtruth": False},
                    {"save_image": False},
                    {"output_folder": "./output_inference"}]
    )
    
    cv_metrics_node = Node(
        package="isp_optimizer",
        executable="cv_metrics",
        name="cv_metrics",
        parameters=[{"confidence_threshold": confidence_threshold}, 
                    {"classes_list": classes_list_}]
    )
    
    cma_es_optimizer_node = Node(
        package="isp_optimizer",
        executable="cma_es_optimizer",
        name="cma_es_optimizer"
    )
    
    ld.add_action(isp_node)
    ld.add_action(data_loader_node)
    ld.add_action(object_detector_node)
    ld.add_action(cv_metrics_node)
    ld.add_action(cma_es_optimizer_node)
    return ld
