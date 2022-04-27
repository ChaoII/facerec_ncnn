cmake_minimum_required(VERSION 3.15)
project(facerec_ncnn)

set(CMAKE_CXX_STANDARD 14)
set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib/cmake/ncnn")
#set(ncnn_DIR "D:/software/ncnn_msvc/debug/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

set(OpenCV_DIR "D:/software/opencv_4.5_msvc/build")
find_package(OpenCV REQUIRED)

include_directories(./)
file(GLOB HEAD ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)


add_executable(${PROJECT_NAME} main.cpp ${SRCS} ${HEAD})
target_link_libraries(${PROJECT_NAME} ncnn ${OpenCV_LIBS})