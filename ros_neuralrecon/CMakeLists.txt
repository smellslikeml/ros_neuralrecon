cmake_minimum_required(VERSION 2.8.3)
project(ros_neuralrecon)

find_package(
    catkin REQUIRED
    COMPONENTS rospy
               rostest
               std_msgs
               geometry_msgs
               sensor_msgs
               visualization_msgs
	       ros_neuralrecon_msgs)

catkin_package()

catkin_python_setup()

catkin_install_python(
    PROGRAMS src/ros_neuralrecon/neural_recon_rosnode.py DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}/ros_neuralrecon)
