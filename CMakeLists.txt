cmake_minimum_required(VERSION 2.8.12)

project(idla)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

option(BUILD_TEST "Determines whether unit tests should be built." OFF)
set(GPU_ARCHITECTURE "sm_30" CACHE INTERNAL "Target GPU architecture for PTX and SASS code generation.")

# Find and build the dlib directory
if (DEFINED DLIB_DIR)
  find_path(DLIB_DIR
    dlib/cmake
    PATHS ${DLIB_DIR} $ENV{DLIB_DIR})

  if (EXISTS ${DLIB_DIR})
    include(${DLIB_DIR}/dlib/cmake)
  else()
    message(FATAL_ERROR "DLIB_DIR must contain 'dlib/cmake'.")
  endif()
else()
  message(FATAL_ERROR "DLIB_DIR must be defined as the path to dlib's root directory.")
endif()

# Check the version of dlib
message(STATUS "dlib Version Found: ${DLIB_VERSION}")
if (${DLIB_VERSION} VERSION_GREATER 19.0.0)
  # If the version is greater than 19.2.0, define a flag that the code will use
  # to implement the new loss layer interface.
  if (${DLIB_VERSION} VERSION_GREATER 19.2.0)
    add_definitions(-DNEW_DLIB_LOSS)
  endif()
else()
  message(FATAL_ERROR "dlib must be at least version 19.0.0. ${DLIB_VERSION} found.")
endif()

# Add HDF5 dependency to this project
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Set source code and required libraries for the main application.
set(source_code
  ${CMAKE_CURRENT_SOURCE_DIR}/src/input.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset.cpp
  )

# Require C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (${DLIB_USE_CUDA})
  # Add CUDA-specific code
  set(source_code
    ${source_code}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/difference_impl.cu
    )

  find_package(CUDA REQUIRED)

  # Set NVCC flags
  list(APPEND CUDA_NVCC_FLAGS "-arch=${GPU_ARCHITECTURE};-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES")
  get_property(dlib_opts TARGET dlib PROPERTY COMPILE_OPTIONS)
  list(APPEND CUDA_NVCC_FLAGS "${dlib_opts}")
  message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

  # Build library code
  cuda_include_directories(${DLIB_DIR})
  cuda_add_library(idla STATIC ${source_code})
  target_link_libraries(idla dlib)
else()
  # Add CUDA-alternative code
  set(source_code
    ${source_code}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/difference_impl.cpp
    )

  add_library(idla STATIC ${source_code})
  target_link_libraries(idla dlib)
endif()

# CUHK03 script
add_executable(run_cuhk03 cuhk03.cpp)
target_link_libraries(run_cuhk03 idla dlib ${HDF5_LIBRARIES})
install(TARGETS run_cuhk03 DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/bin")

if (BUILD_TEST)
  add_subdirectory(test)
endif()
