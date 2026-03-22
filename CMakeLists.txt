cmake_minimum_required(VERSION 3.13)

project(idla LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TEST "Determines whether unit tests should be built." OFF)
set(GPU_ARCHITECTURE "sm_30" CACHE STRING "Target GPU architecture for PTX and SASS code generation.")

# Find dlib
find_package(dlib 20.0 REQUIRED)
message(STATUS "Found dlib version: ${dlib_VERSION}")

# Check for dlib version and set flags if needed
if (${dlib_VERSION} VERSION_GREATER 19.2.0)
  add_definitions(-DNEW_DLIB_LOSS)
endif()

# Find HDF5 (Modern CMake way)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# --- IDLA Library ---
add_library(idla STATIC)

target_sources(idla PRIVATE
  src/input.cpp
  src/dataset.cpp
)

target_include_directories(idla
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Use the HDF5 targets if available, otherwise fallback to variables
if (TARGET HDF5::HDF5)
    target_link_libraries(idla PUBLIC HDF5::HDF5)
endif()
if (TARGET HDF5::HDF5_CPP)
    target_link_libraries(idla PUBLIC HDF5::HDF5_CPP)
endif()

if (NOT TARGET HDF5::HDF5 AND NOT TARGET HDF5::HDF5_CPP)
    target_include_directories(idla PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(idla PUBLIC ${HDF5_LIBRARIES})
endif()

target_link_libraries(idla
  PUBLIC
    dlib::dlib
)

# Handle CUDA if dlib uses it
if (DLIB_USE_CUDA)
  message(STATUS "dlib is using CUDA. Enabling CUDA language.")
  enable_language(CUDA)
  
  target_sources(idla PRIVATE src/difference_impl.cu)
  target_compile_definitions(idla PUBLIC DLIB_USE_CUDA)
  
  # Set CUDA architecture
  set_target_properties(idla PROPERTIES CUDA_ARCHITECTURES 30) # Default to sm_30 for now
  
  # If GPU_ARCHITECTURE is provided, override
  if (GPU_ARCHITECTURE MATCHES "sm_([0-9]+)")
    set_target_properties(idla PROPERTIES CUDA_ARCHITECTURES ${CMAKE_MATCH_1})
  endif()
else()
  target_sources(idla PRIVATE src/difference_impl.cpp)
endif()

# --- CUHK03 Executable ---
add_executable(run_cuhk03)
target_sources(run_cuhk03 PRIVATE cuhk03.cpp)
target_link_libraries(run_cuhk03 PRIVATE idla)

# Installation
install(TARGETS run_cuhk03 DESTINATION bin)

if (BUILD_TEST)
  add_subdirectory(test)
endif()
