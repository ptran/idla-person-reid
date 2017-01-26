cmake_minimum_required(VERSION 2.6)

project(idla)

include_directories(include)

# Find and build the dlib directory
if (DEFINED DLIB_DIR)
  include(${DLIB_DIR}/dlib/cmake)
else()
  message(FATAL_ERROR "DLIB_DIR must be defined as the path to dlib's root directory.")
endif()

# Set source code and required libraries for the main application.
set(source_code)
set(required_libraries dlib)

if (${DLIB_USE_CUDA})
  # Add CUDA-specific code
  set(source_code
    ${CMAKE_CURRENT_SOURCE_DIR}/src/difference.cu
    )

  find_package(CUDA REQUIRED)

  include_directories(${CUDA_INCLUDE_DIRS})
  set(required_libraries
    ${required_libraries}
    ${CUDA_LIBRARIES}
    )
else()
  # Add CUDA-alternative code
endif()

add_executable(${PROJECT_NAME} main.cpp ${source_code})
target_link_libraries(${PROJECT_NAME} ${required_libraries})
