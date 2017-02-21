cmake_minimum_required(VERSION 2.8.4)

project(dtest)

# Set variable for testing suite code
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dlib_testing_suite)
set(test_suite
  ${CMAKE_CURRENT_SOURCE_DIR}/dlib_testing_suite/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dlib_testing_suite/tester.cpp
  )

# Set variable for tests
set(tests
  difference.cpp
  reinterpret.cpp
  )

# Turn on all warnings when using gcc.
if (CMAKE_COMPILER_IS_GNUCXX)
   add_definitions("-W -Wall")
endif()

# Create test executable
add_executable(dtest ${test_suite} ${tests})
target_link_libraries(dtest dlib idla)
install(TARGETS dtest DESTINATION "${CMAKE_SOURCE_DIR}/bin")
