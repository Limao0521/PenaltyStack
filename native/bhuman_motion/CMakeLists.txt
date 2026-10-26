cmake_minimum_required(VERSION 3.21)
project(PenaltyBHumanMotion LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "../.." ABSOLUTE)
set(SRC "${ROOT}/baseline/bhuman/Src")
file(GLOB_RECURSE MOTION CONFIGURE_DEPENDS "${SRC}/Modules/MotionControl/*.cpp")
list(REMOVE_ITEM MOTION "${SRC}/Modules/MotionControl/MotionEngine/MotionEngine.cpp")
list(REMOVE_ITEM MOTION "${SRC}/Modules/MotionControl/WalkingEngine/WalkGenerators/WalkToPoseEngine.cpp"
  "${SRC}/Modules/MotionControl/WalkingEngine/WalkGenerators/InterceptBallProvider.cpp"
  "${SRC}/Modules/MotionControl/WalkingEngine/WalkGenerators/InterceptBallProviderDBHC.cpp")
file(GLOB STREAMING CONFIGURE_DEPENDS "${SRC}/Libs/Streaming/*.cpp")
file(GLOB MATH CONFIGURE_DEPENDS "${SRC}/Libs/Math/*.cpp" "${SRC}/Libs/MathBase/*.cpp")
file(GLOB TOOLS CONFIGURE_DEPENDS "${SRC}/Tools/Motion/*.cpp" "${SRC}/Tools/Math/*.cpp")
file(GLOB REPS CONFIGURE_DEPENDS "${SRC}/Representations/MotionControl/*.cpp" "${SRC}/Representations/Sensing/*.cpp"
    "${SRC}/Representations/Infrastructure/*.cpp" "${SRC}/Representations/Configuration/*.cpp")
list(REMOVE_ITEM REPS "${SRC}/Representations/Infrastructure/CameraImage.cpp" "${SRC}/Representations/Infrastructure/JPEGImage.cpp"
  "${SRC}/Representations/Configuration/FieldDimensions.cpp")
set(SENSORS FsrDataProvider FootSupportProvider GroundContactDetector GyroStateProvider RobotModelProvider
    TorsoMatrixProvider RobotStableStateProvider JointPlayProvider JointPlayTranslationProvider OdometryDataPreviewProvider)
foreach(MODULE IN LISTS SENSORS)
  list(APPEND MOTION "${SRC}/Modules/Sensing/${MODULE}/${MODULE}.cpp")
endforeach()
add_executable(bhuman_motion main.cpp host.cpp kick_factory.cpp keyframe_factory.cpp ${MOTION} ${STREAMING} ${MATH} ${TOOLS} ${REPS}
  "${CMAKE_BINARY_DIR}/generated/MotionEngine.cpp"
  "${CMAKE_BINARY_DIR}/generated/WalkToPoseEngine.cpp"
  "${SRC}/Libs/Framework/Module.cpp" "${SRC}/Libs/Framework/Blackboard.cpp"
  "${CMAKE_BINARY_DIR}/generated/File.cpp" "${SRC}/Libs/Platform/Time.cpp" "${SRC}/Libs/Platform/BHAssert.cpp"
  "${SRC}/Libs/Platform/Memory.cpp"
  "${SRC}/Libs/Debugging/DebugDataTable.cpp" "${SRC}/Libs/Debugging/DebugDataStreamer.cpp"
  "${SRC}/Libs/Debugging/AnnotationManager.cpp" "${SRC}/Libs/Debugging/TimingManager.cpp"
  "${SRC}/Libs/Debugging/ColorRGBA.cpp" "${SRC}/Libs/Debugging/DebugRequest.cpp" "${SRC}/Libs/Debugging/DebugDrawings.cpp"
  "${SRC}/Libs/RobotParts/FootShape.cpp"
  "${SRC}/Tools/Modeling/BallPhysics.cpp" "${SRC}/Tools/Modeling/Obstacle.cpp"
  "${SRC}/Representations/Modeling/BallModel.cpp" "${SRC}/Representations/Modeling/ObstacleModel.cpp"
  "${SRC}/Representations/Modeling/RobotPose.cpp"
  "${SRC}/Representations/Perception/ImagePreprocessing/CameraMatrix.cpp"
  "${SRC}/Modules/Sensing/FallDownStateDetector/FallDownStateProvider.cpp")
# generated/ va primero: contiene las adaptaciones de plataforma (p. ej.
# Math/Range.h) que sustituyen a los originales sin modificar sus bytes.
target_include_directories(bhuman_motion PRIVATE "${CMAKE_BINARY_DIR}/generated" "${SRC}" "${SRC}/Libs"
  "${SRC}/Modules/MotionControl/MotionEngine" "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen-3.4.0"
  "${SRC}/Modules/MotionControl/WalkingEngine/WalkGenerators"
  "${CMAKE_CURRENT_SOURCE_DIR}/dependencies")
target_compile_definitions(bhuman_motion PRIVATE BHUMAN_BASELINE_DIR="${ROOT}/baseline/bhuman")
if(MSVC)
  target_compile_definitions(bhuman_motion PRIVATE WINDOWS NOMINMAX _CRT_SECURE_NO_WARNINGS)
  target_compile_options(bhuman_motion PRIVATE /bigobj /MP4 /Zc:preprocessor /wd4267 /wd4244 /wd4305 /wd4996)
else()
  # La fuente original ramifica en Platform/*.cpp con `defined LINUX`.
  target_compile_definitions(bhuman_motion PRIVATE LINUX)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  target_compile_options(bhuman_motion PRIVATE -Wno-conversion -Wno-sign-conversion
    -Wno-unused-parameter -Wno-unused-variable -Wno-missing-field-initializers
    -Wno-unused-but-set-variable -Wno-deprecated-enum-enum-conversion -fno-strict-aliasing)
endif()
target_precompile_headers(bhuman_motion PRIVATE <cmath> <algorithm> <limits> <unordered_map>
  "${SRC}/Libs/Math/BHMath.h" "${SRC}/Libs/Math/Eigen.h")
# Range.h se precarga desde generated/: es la adaptación de plataforma que
# también resuelven todos los `#include "Math/Range.h"` (generated va primero).
target_precompile_headers(bhuman_motion PRIVATE "${SRC}/Libs/Math/RingBufferWithSum.h" "${CMAKE_BINARY_DIR}/generated/Math/Range.h"
  "${SRC}/Libs/Math/Pose3f.h" "${SRC}/Libs/Math/Geometry.h" "${SRC}/Libs/Debugging/DebugDrawings.h"
  "${SRC}/Libs/Framework/Module.h" "${SRC}/Libs/Streaming/MessageQueue.h")
find_package(Threads REQUIRED)
target_link_libraries(bhuman_motion PRIVATE Threads::Threads)
if(MSVC)
  target_link_libraries(bhuman_motion PRIVATE winmm)
endif()
