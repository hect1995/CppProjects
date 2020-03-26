cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)
set(Torch_DIR "/Users/hectoresteban/libtorch/share/cmake/torch")
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.2.0_3/lib/cmake/opencv4")
find_package( OpenCV REQUIRED )
find_package(Torch REQUIRED)

#add_executable(learn main.cpp)
add_executable(pair from_python.cpp)

target_link_libraries(pair ${OpenCV_LIBS})
target_link_libraries(pair "${TORCH_LIBRARIES}")

set_property(TARGET pair PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dcgan
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:dcgan>)
endif (MSVC)
