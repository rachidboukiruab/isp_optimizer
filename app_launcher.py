from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    data_loader_node = Node(
        package="isp_optimizer",
        executable="data_loader",
        name="data_loader",
        parameters=[{"annotations_folder_path": "//home/rachid/raw_dataset/test/"}, 
                    {"classes_list": ["car"]},
                    {"publish_frequency": 1.0}]
    )

    ld.add_action(data_loader_node)
    return ld
