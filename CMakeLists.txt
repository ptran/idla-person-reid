cmake_minimum_required(VERSION 3.18)

project(idla LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TEST "Determines whether unit tests should be built." OFF)
set(GPU_ARCHITECTURE "86" CACHE STRING "Target GPU architecture (e.g., 86 for RTX 30-series).")

# --- Find Dependencies ---

# Find dlib
find_package(dlib 20.0 REQUIRED)
message(STATUS "Found dlib version: ${dlib_VERSION}")

# Find HDF5
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

# Apply dlib-specific definitions
if (${dlib_VERSION} VERSION_GREATER 19.2.0)
  target_compile_definitions(idla PUBLIC NEW_DLIB_LOSS)
endif()

# Link HDF5
if (TARGET HDF5::HDF5_CPP)
  target_link_libraries(idla PUBLIC HDF5::HDF5_CPP)
elseif (TARGET HDF5::HDF5)
  target_link_libraries(idla PUBLIC HDF5::HDF5)
endif()

# Link dlib
target_link_libraries(idla PUBLIC dlib::dlib)

# --- Modern CUDA Detection (CMake 3.18+) ---
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}. Enabling CUDA.")
  enable_language(CUDA)
  
  # Find toolkit libraries (like cudart)
  find_package(CUDAToolkit REQUIRED)
  
  target_sources(idla PRIVATE src/difference_impl.cu)
  target_compile_definitions(idla PUBLIC DLIB_USE_CUDA)
  
  # Link to CUDA runtime (provides headers and libraries)
  target_link_libraries(idla PUBLIC CUDA::cudart)
  
  # Modern architecture handling (CMake 3.18+)
  set_target_properties(idla PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCHITECTURE})
  message(STATUS "Set CUDA_ARCHITECTURES to ${GPU_ARCHITECTURE}")
else()
  message(STATUS "CUDA compiler NOT found. Using CPU implementation.")
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
