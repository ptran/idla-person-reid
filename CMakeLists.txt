cmake_minimum_required(VERSION 3.1)

project(idla)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

option(BUILD_TEST "Determines whether unit tests should be built." OFF)
set(GPU_ARCHITECTURE "sm_52" CACHE INTERNAL "Target GPU architecture for PTX and SASS code generation.")

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

  # If the version is less than 19.4.0, make an ALIAS target dlib::dlib with the
  # target produced by dlib's in-project build.
  if (${DLIB_VERSION} VERSION_LESS 19.4.0)
    add_library(dlib::dlib ALIAS dlib)
  endif()
else()
  message(FATAL_ERROR "dlib must be at least version 19.0.0. ${DLIB_VERSION} found.")
endif()

# Set source code and required libraries for the main application.
set(source_code
  ${CMAKE_CURRENT_SOURCE_DIR}/src/input.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset.cpp
  )

# Require C++11
set(CMAKE_CXX_STANDARD 11)

if (${DLIB_USE_CUDA})
  add_definitions("-DDLIB_USE_CUDA")

  # Add CUDA-specific code
  set(source_code
    ${source_code}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/difference_impl.cu
    )

  find_package(CUDA REQUIRED)

  # Set NVCC flags
  list(APPEND CUDA_NVCC_FLAGS "-arch=${GPU_ARCHITECTURE};-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES")
  message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

  # Build library code
  cuda_include_directories(${DLIB_DIR})
  cuda_add_library(idla STATIC ${source_code})
  target_link_libraries(idla dlib::dlib)
else()
  # Add CUDA-alternative code
  set(source_code
    ${source_code}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/difference_impl.cpp
    )

  add_library(idla STATIC ${source_code})
  target_link_libraries(idla dlib::dlib)
endif()

# CUHK03 script
add_executable(run_cuhk03 src/apps/cuhk03.cpp)
target_link_libraries(run_cuhk03 idla dlib::dlib)
install(TARGETS run_cuhk03 DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/bin")

if (BUILD_TEST)
  add_subdirectory(test)
endif()
