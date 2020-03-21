cmake_minimum_required(VERSION 3.13)
project(OPENCV)

set(CMAKE_CXX_STANDARD 17)

include_directories("/Users/hectoresteban/Documents/C++/plog-master/include")

find_package( OpenCV REQUIRED )

#add_executable(learn main.cpp)
add_executable(pair main.cpp processing.cpp)
target_link_libraries(pair ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(pair /Users/hectoresteban/Documents/C++/plog-master/include/plog)
