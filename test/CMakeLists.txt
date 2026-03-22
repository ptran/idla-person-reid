cmake_minimum_required(VERSION 3.13)

project(dtest LANGUAGES CXX)

# Create test executable
add_executable(dtest)

target_sources(dtest PRIVATE
  dlib_testing_suite/main.cpp
  dlib_testing_suite/tester.cpp
  difference.cpp
  reinterpret.cpp
  multiclass_loss.cpp
)

target_include_directories(dtest PRIVATE
  dlib_testing_suite
)

target_link_libraries(dtest PRIVATE
  idla
  dlib::dlib
)

# Installation
install(TARGETS dtest DESTINATION bin)
