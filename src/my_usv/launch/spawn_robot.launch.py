import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('my_usv')
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot.urdf')

    # --- ARGOMENTI DA TERMINALE ---
    world_arg = DeclareLaunchArgument('world', default_value=os.path.join(pkg_share, 'worlds', 'labirinto_9a.world'))
    x_arg = DeclareLaunchArgument('x', default_value='0.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0', description='Rotazione (Yaw) in radianti')

    world_config = LaunchConfiguration('world')
    x_config = LaunchConfiguration('x')
    y_config = LaunchConfiguration('y')
    yaw_config = LaunchConfiguration('yaw')
    # ------------------------------

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world_config}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description', 
            '-entity', 'usv_robot', 
            '-x', x_config, 
            '-y', y_config, 
            '-z', '0.5',
            '-Y', yaw_config  # <-- LA MAGIA E' QUI (Y maiuscola = Yaw)
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg, x_arg, y_arg, yaw_arg,
        gazebo_launch, robot_state_publisher, spawn_entity
    ])
