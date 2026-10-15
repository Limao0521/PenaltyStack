cmake_minimum_required(VERSION 3.17)
project(CopaPolicyTests LANGUAGES CXX)

# Host-side parity test of the header that runs on the NAO (no B-Human dependency).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(copa_policy_tests main.cpp)
target_include_directories(copa_policy_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../robot/Src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../bhuman_motion/dependencies")
if(MSVC)
  target_compile_options(copa_policy_tests PRIVATE /W4 /permissive- /EHsc /bigobj)
else()
  target_compile_options(copa_policy_tests PRIVATE -Wall -Wextra -Wconversion -Wno-sign-conversion)
endif()
