cmake_minimum_required(VERSION 3.13)
project(BackgroundEstimation)


set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BackgroundEstimation main.cpp calculate_median.cpp)
target_link_libraries(BackgroundEstimation ${OpenCV_LIBS})
