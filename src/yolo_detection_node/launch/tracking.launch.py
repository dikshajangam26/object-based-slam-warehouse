import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. Define the Tracking Node (YOLO + ByteTrack)
    tracking_node = Node(
        package='yolo_detection_node',
        executable='tracking_node',
        name='tracking_node',
        output='screen',
        parameters=[
            {'use_sim_time': True} 
            # {'max_history': 30} # You can pass parameters here later
        ]
    )

    # 2. Define the 3D Reconstruction Node
    reconstruction_node = Node(
        package='yolo_detection_node',
        executable='reconstruction_node',
        name='reconstruction_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ],
        remappings=[
            ('/tf', '/cpr_r100_0000/tf'),
            ('/tf_static', '/cpr_r100_0000/tf_static')
        ]
    )
    
    # 3. Global Mapping Node (Dedicated mapping_node package)
    mapping_node = Node(
        package='mapping_node',
        executable='mapping_node',
        name='mapping_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('/tf', '/cpr_r100_0000/tf'),
            ('/tf_static', '/cpr_r100_0000/tf_static')
        ]
    )
    
    # 3. Return them both to the ROS 2 Launch execution engine
    return LaunchDescription([
        tracking_node,
        reconstruction_node,
        mapping_node
    ])