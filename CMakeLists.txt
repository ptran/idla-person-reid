cmake_minimum_required(VERSION 2.6)

project(idla)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find and build the dlib directory
if (DEFINED DLIB_DIR)
  include(${DLIB_DIR}/dlib/cmake)
else()
  message(FATAL_ERROR "DLIB_DIR must be defined as the path to dlib's root directory.")
endif()

# Set source code and required libraries for the main application.
set(source_code
  ${CMAKE_CURRENT_SOURCE_DIR}/src/input.cpp
  )
set(required_libraries dlib)

# Require C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (${DLIB_USE_CUDA})
  # Add CUDA-specific code
  set(source_code
    ${source_code}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/difference_impl.cu
    )
  find_package(CUDA REQUIRED)
  # Following line taken from dlib's CMakeLists.txt
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES")
  cuda_add_executable(${PROJECT_NAME} main.cpp ${source_code})
  target_link_libraries(${PROJECT_NAME} ${required_libraries})
else()
  # Add CUDA-alternative code
endif()
