cmake_minimum_required(VERSION 3.21)
project(PenaltyBHumanVision LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "../.." ABSOLUTE)
set(ORIGINAL "${ROOT}/heritage/bhuman_robot_perception")
set(DEPS "${ROOT}/native/bhuman_motion/build/vision_deps")
add_executable(bhuman_vision main.cpp)
target_include_directories(bhuman_vision PRIVATE
  "${ORIGINAL}/Util/onnxruntime/include"
  "${ROOT}/native/bhuman_motion/dependencies"
  "${ROOT}/native/bhuman_motion/dependencies/eigen-3.4.0"
  "${ROOT}/native/bhuman_motion/build/generated/vision")
if(WIN32)
  target_compile_definitions(bhuman_vision PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
  target_compile_options(bhuman_vision PRIVATE /W4 /wd4244 /wd4267)
  target_link_libraries(bhuman_vision PRIVATE "${DEPS}/onnxruntime.lib")
  add_custom_command(TARGET bhuman_vision POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DEPS}/onnxruntime.dll" "$<TARGET_FILE_DIR:bhuman_vision>")
else()
  target_compile_options(bhuman_vision PRIVATE -Wall -Wextra -msse2)
  target_link_libraries(bhuman_vision PRIVATE "${DEPS}/libonnxruntime.so.1.10.0")
  set_target_properties(bhuman_vision PROPERTIES BUILD_RPATH "${DEPS};$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()
