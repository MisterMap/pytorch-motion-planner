cmake_minimum_required(VERSION 2.8.12)
project(pytorch_motion_planner)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        nav_msgs
        geometry_msgs
        message_generation
        )

catkin_python_setup()

catkin_package()

install(PROGRAMS
        scripts/pytorch_motion_planner_node.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
