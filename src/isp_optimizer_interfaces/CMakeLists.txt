cmake_minimum_required(VERSION 3.8)
project(isp_optimizer_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} 
  "msg/BoundingBox.msg"
  "msg/ListBoundingBox.msg"
  "msg/ComputerVisionMetrics.msg"
  "msg/Inference.msg" 
  "srv/RunInference.srv"
  "srv/RunEvaluation.srv"
  "srv/GetMetricResults.srv"
  DEPENDENCIES sensor_msgs
  )

ament_export_dependencies(rosidl_default_runtime)

ament_package()
