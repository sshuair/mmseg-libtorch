cmake_minimum_required(VERSION 3.14)

project(mmseg)

set(EXECUTABLE_NAME mmseg)

find_package(Torch REQUIRED )
find_package(OpenCV REQUIRED )

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
        src/utils.cpp
        include/utils.h
)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Torch_INCLUDE_DIRS})

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBRARIES})



set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)