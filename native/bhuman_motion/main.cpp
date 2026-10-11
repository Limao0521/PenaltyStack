// JSONL host of the genuine B-Human MODULE implementations and blackboard.
#include "json.hpp"
#include "provenance.h"
#include "Framework/Blackboard.h"
#include "Framework/Settings.h"
#include "Streaming/FunctionList.h"
#include "Streaming/InStreams.h"
#include "Streaming/Global.h"
#include "Debugging/AnnotationManager.h"
#include "Debugging/DebugDataTable.h"
#include "Debugging/DebugRequest.h"
#include "Debugging/DebugDrawings.h"
#include "Debugging/DebugDrawings3D.h"
#include "Debugging/TimingManager.h"
#include "Platform/File.h"
#include "Math/Rotation.h"
#include "Math/Random.h"
#include "Tools/Motion/Transformation.h"
#include "Modules/MotionControl/MotionEngine/MotionEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkingEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkKickEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkLibs/WalkLearnerProvider.h"
#include "Modules/MotionControl/WalkingEngine/WalkGenerators/WalkAtSpeedEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkGenerators/WalkToPoseEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkGenerators/WalkToBallEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkGenerators/WalkToBallAndKickEngine.h"
#include "Modules/MotionControl/KickEngine/KickEngineParameters.h"
#include "Representations/MotionControl/KickGenerator.h"
#include "Modules/MotionControl/WalkingEngine/WalkGenerators/DribbleEngine.h"
#include "Modules/MotionControl/WalkingEngine/WalkGenerators/ReplayWalkRequestProvider.h"
#include "Modules/MotionControl/HeadMotionEngine/HeadMotionEngine.h"
#include "Modules/MotionControl/ArmKeyFrameEngine/ArmKeyFrameEngine.h"
#include "factories.h"
#include "Representations/Configuration/StaticJointPoses.h"
#include "Representations/MotionControl/KeyframeMotionParameters.h"
#include "Modules/MotionControl/FallEngine/FallEngine.h"
#include "Modules/MotionControl/EnergySaving/EnergySavingProvider.h"
#include "Modules/MotionControl/PointAtEngine/PointAtEngine.h"
#include "Modules/Sensing/FsrDataProvider/FsrDataProvider.h"
#include "Modules/Sensing/FootSupportProvider/FootSupportProvider.h"
#include "Modules/Sensing/GroundContactDetector/GroundContactDetector.h"
#include "Modules/Sensing/GyroStateProvider/GyroStateProvider.h"
#include "Modules/Sensing/RobotModelProvider/RobotModelProvider.h"
#include "Modules/Sensing/TorsoMatrixProvider/TorsoMatrixProvider.h"
#include "Modules/Sensing/RobotStableStateProvider/RobotStableStateProvider.h"
#include "Modules/Sensing/JointPlayProvider/JointPlayProvider.h"
#include "Modules/Sensing/JointPlayTranslationProvider/JointPlayTranslationProvider.h"
#include "Modules/Sensing/OdometryDataPreviewProvider/OdometryDataPreviewProvider.h"
#include "Modules/Sensing/FallDownStateDetector/FallDownStateProvider.h"
#include "Representations/Infrastructure/SensorData/JointSensorData.h"
#include <iostream>
#include <memory>
#include <filesystem>
#include <functional>
#undef action

using json = nlohmann::json;
static constexpr const char* revision = "2360ab1100ad28f70e190c893f6b095651ff0bbc";

// Original Global explicitly grants this class access for thread initialization.
// No original ThreadFrame is linked: the standalone process owns one Blackboard.
class ThreadFrame
{
public:
  static void set(Settings& settings, AnnotationManager& annotations, MessageQueue& out,
                  DebugRequestTable& requests, DebugDataTable& data, DrawingManager& drawing,
                  DrawingManager3D& drawing3D, TimingManager& timing)
  {
    Global::theSettings = &settings;
    Global::theAnnotationManager = &annotations;
    Global::theDebugOut = &out;
    Global::theDebugRequestTable = &requests;
    Global::theDebugDataTable = &data;
    Global::theDrawingManager = &drawing;
    Global::theDrawingManager3D = &drawing3D;
    Global::theTimingManager = &timing;
  }
};

class Runtime
{
  Blackboard board;
  Settings settings{"Nao", "Nao", 1, Settings::blue, Settings::red, 1, "Default", "Default", 0};
  AnnotationManager annotations;
  MessageQueue debug;
  DebugRequestTable requests;
  DebugDataTable data;
  DrawingManager drawings;
  DrawingManager3D drawings3D;
  TimingManager timing;
  std::map<std::string, NativeModule> modules;
  std::vector<std::string> sensorOrder, generatorOrder;
  bool constructed = false;
  unsigned lastKick = 0;
  int64_t lastSequence = -1;
  int64_t lastTime = -12;
  bool keeperMotionLatched = false;
  json keeperMotionRequest = nullptr, keeperMotionSummary = nullptr;
  NativeKeeperMotionParameters keeperMotionParameters;

  template<class T> T& rep(const char* name) { return static_cast<T&>(board[name]); }
  template<class T> T& allocate(const char* name) { return board.alloc<T>(name); }
  template<class T> void config(const char* name, const char* file)
  {
    auto& value = allocate<T>(name);
    InMapFile stream(file);
    if(!stream.exists()) throw std::runtime_error(std::string("Missing configuration: ") + file);
    stream >> value;
  }
  template<class M> void add(const std::string& name, std::vector<std::string>& order)
  {
    std::unique_ptr<Streamable> instance;
    if constexpr(std::is_same_v<M,WalkingEngine>) instance.reset(static_cast<WalkingEngineBase*>(new M));
    else instance.reset(new M);
    modules.emplace(name, NativeModule{std::move(instance), M::getModuleInfo()});
    order.push_back(name);
  }
  void update(const std::string& name, const char* only = nullptr)
  {
    auto& module = modules.at(name);
    for(const auto& info : module.info)
      if(info.update && (!only || std::string(info.representation) == only))
        info.update(*module.instance);
  }
  void construct()
  {
#define SENSOR(M) add<M>(#M, sensorOrder); update(#M)
    SENSOR(FsrDataProvider); SENSOR(FootSupportProvider); SENSOR(GroundContactDetector);
    SENSOR(GyroStateProvider); SENSOR(RobotModelProvider); SENSOR(TorsoMatrixProvider);
    SENSOR(FallDownStateProvider); SENSOR(JointPlayProvider); SENSOR(JointPlayTranslationProvider);
    SENSOR(RobotStableStateProvider); SENSOR(OdometryDataPreviewProvider);
#undef SENSOR
#define ENGINE(M) add<M>(#M, generatorOrder); update(#M)
    ENGINE(EnergySavingProvider);
    modules.emplace("KeyframeMotionEngine",makeNativeKeyframeMotionEngine());
    generatorOrder.push_back("KeyframeMotionEngine"); update("KeyframeMotionEngine");
    modules.emplace("KickEngine",makeNativeKickEngine());
    generatorOrder.push_back("KickEngine"); update("KickEngine");
    ENGINE(WalkLearnerProvider); ENGINE(WalkingEngine); ENGINE(WalkKickEngine);
    ENGINE(WalkAtSpeedEngine); ENGINE(WalkToPoseEngine); ENGINE(WalkToBallEngine);
    ENGINE(WalkToBallAndKickEngine); ENGINE(DribbleEngine); ENGINE(ReplayWalkRequestProvider);
    ENGINE(HeadMotionEngine); ENGINE(ArmKeyFrameEngine); ENGINE(PointAtEngine); ENGINE(FallEngine);
#undef ENGINE
    std::vector<std::string> unused;
    add<MotionEngine>("MotionEngine", unused);
    constructed = true;
  }
  static std::vector<float> vector(const json& value, size_t count, float bound)
  {
    if(!value.is_array() || value.size() != count) throw std::runtime_error("Wrong sensor vector size");
    std::vector<float> result;
    for(const auto& element : value)
    {
      if(!element.is_number()) throw std::runtime_error("Sensor must be numeric");
      const double x = element.get<double>();
      if(!std::isfinite(x) || std::abs(x) > bound) throw std::runtime_error("Nonfinite or out-of-range sensor");
      result.push_back(static_cast<float>(x));
    }
    return result;
  }
  static float scalar(const json& request, const char* name, float fallback, float bound)
  {
    if(!request.contains(name)) return fallback;
    if(!request[name].is_number()) throw std::runtime_error("Request scalar must be numeric");
    float x = request[name].get<float>();
    if(!std::isfinite(x) || std::abs(x) > bound) throw std::runtime_error("Invalid request scalar");
    return x;
  }
  static NativeKeeperMotionParameters keeperSkill(const json& skill)
  {
    if(!skill.is_object() || skill.size() != 2 || !skill.contains("version") ||
       skill.at("version") != "keeper_motion_skill_v1" || !skill.contains("parameters"))
      throw std::runtime_error("Invalid keeper_motion_skill_v1 contract");
    const auto& values = skill.at("parameters");
    if(!values.is_object() || values.size() != 4)
      throw std::runtime_error("Keeper motion skill requires exactly four parameters");
    const auto number = [&](const char* name, double low, double high)
    {
      if(!values.contains(name) || !values.at(name).is_number())
        throw std::runtime_error("Keeper motion skill parameter must be numeric");
      const double value = values.at(name).get<double>();
      if(!std::isfinite(value) || value < low || value > high)
        throw std::runtime_error("Keeper motion skill parameter outside bounds");
      return static_cast<float>(value);
    };
    return {number("guard_time_scale",.85,1.15), number("jump_time_scale",.85,1.15),
            number("arm_spread_rad",-.08,.12), number("guard_knee_offset_rad",0.,.10)};
  }

public:
  std::string trial, role;
  Runtime(const json& init) : trial(init.at("trial").get<std::string>()), role(init.at("role").get<std::string>())
  {
    if(!init.contains("protocol") || !init["protocol"].is_number_integer() || init["protocol"] != 1 ||
       !init.contains("seed") || !init["seed"].is_number_integer() || init["seed"].get<int64_t>() < 0 || init["seed"].get<int64_t>() > 4294967295LL || trial.empty() || trial.size() > 128 ||
       (role != "striker" && role != "keeper")) throw std::runtime_error("Invalid INIT");
    ThreadFrame::set(settings, annotations, debug, requests, data, drawings, drawings3D, timing);
    File::setSearchPath(settings.getSearchPath());
    Random::getGenerator().seed(init.value("seed", 0u));
#define CFG(T, file) config<T>(#T, file)
    CFG(RobotDimensions, "robotDimensions.cfg"); CFG(MassCalibration, "massCalibration.cfg");
    CFG(JointLimits, "jointLimits.cfg"); CFG(StiffnessSettings, "stiffnessSettings.cfg");
    CFG(FootOffset, "footOffset.cfg"); CFG(KickInfo, "kickInfo.cfg");
    CFG(StaticJointPoses, "staticJointPoses.cfg"); CFG(KeyframeMotionParameters, "keyframeMotionParameters.cfg");
    CFG(WalkModifier, "walkModifier.cfg"); CFG(HeadLimits, "headLimits.cfg");
    CFG(BallSpecification, "ballSpecification.cfg"); CFG(CameraIntrinsics, "cameraIntrinsics.cfg");
    CFG(CameraCalibration, "cameraCalibration.cfg"); CFG(DamageConfigurationBody, "damageConfigurationBody.cfg");
#undef CFG
    // Upstream configured 65 mm as radius; this competition uses a 65 mm
    // diameter ball. The world supplies its actual geometry at initialization.
    auto& ballSpecification = rep<BallSpecification>("BallSpecification");
    ballSpecification.radius = scalar(init,"ball_radius_mm",32.5f,200.f);
    if(ballSpecification.radius <= 0.f) throw std::runtime_error("Ball radius must be positive");
#define ALLOC(T) allocate<T>(#T)
    ALLOC(FrameInfo); ALLOC(CognitionFrameInfo); ALLOC(UpperFrameInfo); ALLOC(LowerFrameInfo);
    ALLOC(JointAngles); ALLOC(JointSensorData); ALLOC(JointRequest); ALLOC(InertialData); ALLOC(InertialSensorData);
    ALLOC(FsrSensorData); ALLOC(HeadMotionRequest); ALLOC(MotionRequest); ALLOC(ArmMotionRequest);
    ALLOC(JointAnglePred); ALLOC(GyroOffset).offsetCheckFinished = true;
    ALLOC(FallDownState).state = FallDownState::upright;
#undef ALLOC
  }

  json step(const json& message)
  {
    if(message.at("trial").get<std::string>() != trial) throw std::runtime_error("Wrong trial");
    if(!message.at("sequence").is_number_integer() || !message.at("time_ms").is_number_integer())
      throw std::runtime_error("Sequence and time must be integers");
    const int64_t sequence = message.at("sequence"), time = message.at("time_ms");
    if(sequence != lastSequence + 1 || time < 0 || time > 86400000 || (lastSequence >= 0 && time != lastTime + 12))
      throw std::runtime_error("Sequence or motion clock is discontinuous");
    const json& sensors = message.at("sensors");
    auto angles = vector(sensors.at("angles"), 26, 7.f);
    auto velocities = vector(sensors.at("velocity"), 26, 100.f);
    auto gyro = vector(sensors.at("gyro"), 3, 100.f);
    auto acc = vector(sensors.at("acc"), 3, 1000.f);
    auto orientation = vector(sensors.at("orientation"), 4, 1.01f);
    auto fsr = vector(sensors.at("fsr"), 8, 100.f);
    Quaternionf quaternion(orientation[0], orientation[1], orientation[2], orientation[3]);
    if(std::abs(quaternion.norm() - 1.f) > .01f) throw std::runtime_error("Orientation is not a unit quaternion");
    quaternion.normalize();
    for(float pressure : fsr) if(pressure < 0.f) throw std::runtime_error("Negative foot pressure");
    auto& frame = rep<FrameInfo>("FrameInfo"); frame.time = 100000 + static_cast<unsigned>(time);
    rep<CognitionFrameInfo>("CognitionFrameInfo").time = frame.time;
    rep<UpperFrameInfo>("UpperFrameInfo").time = frame.time;
    rep<LowerFrameInfo>("LowerFrameInfo").time = frame.time;
    auto& measured = rep<JointAngles>("JointAngles"); auto& raw = rep<JointSensorData>("JointSensorData");
    FOREACH_ENUM(Joints::Joint, joint) { measured.angles[joint] = angles[joint]; raw.angles[joint] = angles[joint]; }
    measured.timestamp = raw.timestamp = frame.time;
    if(!constructed)
    {
      auto& initial = rep<JointRequest>("JointRequest");
      initial.angles = measured.angles;
      initial.stiffnessData.stiffnesses.fill(70);
    }
    raw.currents.fill(0); raw.temperatures.fill(35);
    auto& predicted = rep<JointAnglePred>("JointAnglePred");
    static constexpr Joints::Joint mapping[] = {Joints::rAnkleRoll,Joints::rAnklePitch,Joints::rKneePitch,Joints::rHipPitch,Joints::rHipRoll,
       Joints::lKneePitch,Joints::lAnkleRoll,Joints::lAnklePitch,Joints::lHipPitch,Joints::lHipRoll,Joints::lHipYawPitch};
    FOREACH_ENUM(JointAnglePred::Joint, joint) predicted.angles[joint] = angles[mapping[joint]];
    auto& inertial = rep<InertialData>("InertialData");
    Vector2a oldAngles = inertial.angle.head<2>();
    inertial.gyro.x() = gyro[0]; inertial.gyro.y() = gyro[1]; inertial.gyro.z() = gyro[2];
    inertial.acc = Vector3f(acc[0], acc[1], acc[2]);
    inertial.orientation3D = quaternion;
    // Match InertialDataProvider exactly: tilt is a horizontal angle-axis
    // vector. Euler decomposition is invalid for the get-up/fall contract.
    inertial.orientation2D = Rotation::removeZRotation(quaternion);
    inertial.angle << Rotation::AngleAxis::pack(AngleAxisf(inertial.orientation2D)).head<2>().cast<Angle>(), 0_deg;
    inertial.angleChange = inertial.angle.head<2>() - oldAngles;
    rep<InertialSensorData>("InertialSensorData") = inertial;
    auto& pressures = rep<FsrSensorData>("FsrSensorData");
    FOREACH_ENUM(Legs::Leg, leg)
    {
      pressures.totals[leg] = 0;
      FOREACH_ENUM(FsrSensors::FsrSensor, sensor) { pressures.pressures[leg][sensor] = fsr[4 * leg + sensor]; pressures.totals[leg] += fsr[4 * leg + sensor]; }
    }
    const json& request = message.at("request");
    auto& motion = rep<MotionRequest>("MotionRequest");
    const std::string action = request.at("motion");
    if(request.contains("keeper_motion_skill") && (role != "keeper" || action != "block"))
      throw std::runtime_error("Keeper motion skill is only valid for keeper block requests");
    bool applyKeeperMotion = false;
    motion = MotionRequest();
    motion.walkSpeed = Pose2f(scalar(request,"turn",0,3),scalar(request,"forward",0,500),scalar(request,"lateral",0,300));
    if(action == "stand" || action == "getup") motion.motion = MotionRequest::stand;
    else if(action == "walk") motion.motion = MotionRequest::walkAtAbsoluteSpeed;
    else if(action == "kick")
    {
      motion.motion = MotionRequest::walkToBallAndKick;
      motion.walkSpeed = Pose2f(.5f,.5f,.5f);
      std::string type = request.value("kick_type", "forwardFastLeft");
      if(type != "forwardFastLeft" && type != "forwardFastRight") throw std::runtime_error("Unknown kick type");
      motion.kickType = type == "forwardFastLeft" ? KickInfo::forwardFastLeft : KickInfo::forwardFastRight;
      motion.kickLength = scalar(request,"kick_length_mm",2500.f,3500.f);
      if(motion.kickLength < 1500.f) throw std::runtime_error("Kick length must be 1500..3500 mm");
      auto ball = vector(request.at("ball"),2,10000);
      motion.ballEstimate.position = Vector2f(ball[0],ball[1]);
      motion.ballEstimate.velocity = Vector2f::Zero();
      motion.ballEstimateTimestamp = motion.ballTimeWhenLastSeen = frame.time;
      motion.odometryData = constructed ? rep<OdometryDataPreview>("OdometryDataPreview") : Pose2f();
    }
    else if(action == "block")
    {
      motion.motion = MotionRequest::dive;
      const std::string side = request.value("block_side", "center");
      if(side == "prepare" && request.contains("keeper_motion_skill"))
        throw std::runtime_error("Keeper preparation does not support a trained motion profile");
      if(side != "prepare" && role == "keeper")
      {
        const json profile = request.contains("keeper_motion_skill") ? request.at("keeper_motion_skill") : json(nullptr);
        if(keeperMotionLatched)
        {
          if(profile != keeperMotionRequest)
            throw std::runtime_error("Keeper motion skill cannot change or be omitted after the first block");
        }
        else
        {
          keeperMotionLatched = true;
          keeperMotionRequest = profile;
          if(request.contains("keeper_motion_skill"))
          {
            keeperMotionParameters = keeperSkill(profile);
            applyKeeperMotion = true;
          }
        }
      }
      // Each request selects an existing upstream motion. The original
      // KeyframeMotionRequest mapping owns the keyframes and their mirrors.
      if(side == "center") motion.diveRequest = MotionRequest::Dive::squatArmsBackLeft;
      else if(side == "center_right") motion.diveRequest = MotionRequest::Dive::squatArmsBackRight;
      else if(side == "wide_left") motion.diveRequest = MotionRequest::Dive::squatWideArmsBackLeft;
      else if(side == "wide_right") motion.diveRequest = MotionRequest::Dive::squatWideArmsBackRight;
      else if(side == "left") motion.diveRequest = MotionRequest::Dive::jumpLeft;
      else if(side == "right") motion.diveRequest = MotionRequest::Dive::jumpRight;
      else if(side == "prepare") motion.diveRequest = MotionRequest::Dive::prepare;
      else throw std::runtime_error("Unknown block side");
    }
    else throw std::runtime_error("Unknown motion request");
    auto& head = rep<HeadMotionRequest>("HeadMotionRequest");
    head.pan = scalar(request,"head_yaw",0,2.1f); head.tilt = scalar(request,"head_pitch",0,.8f);
    if(!constructed) construct();
    if(applyKeeperMotion)
      keeperMotionSummary = json::parse(applyNativeKeeperMotionSkill(modules.at("KeyframeMotionEngine"),
          keeperMotionParameters, rep<JointLimits>("JointLimits")));
    for(const auto& name : sensorOrder) update(name);
    for(const auto& name : generatorOrder) update(name);
    // The 100 ms right strike has 0.2 rad knee tracking lag in this plant.
    // Give the same original dynamic trajectory 200 ms to execute instead.
    // Keep the working left strike and all original trajectory endpoints.
    // Original balancing, IK, mirror and support foot stay in
    // KickEngine. This adapter never changes measured joints/root/ball state.
    auto& kickGenerator = rep<KickGenerator>("KickGenerator");
    const auto originalKick = kickGenerator.createPhase;
    kickGenerator.createPhase = [originalKick](const KickRequest& request,const MotionPhase& lastPhase)
    {
      KickRequest calibrated=request;
      if(request.kickMotionType==KickRequest::kickForwardFast && !request.mirror && request.calcDynPoints)
      {
        const auto originalPoints=request.calcDynPoints;
        calibrated.calcDynPoints = [originalPoints](const int phase)
        {
          auto points=originalPoints(phase);
          for(auto& point:points)
            if(point.limb==Phase::rightFootTra && point.phaseNumber==3)
              point.duration=200;
          return points;
        };
      }
      return originalKick(calibrated,lastPhase);
    };
    update("MotionEngine", "JointRequest"); update("MotionEngine", "MotionInfo"); update("MotionEngine", "OdometryData");
    update("MotionEngine", "ArmMotionInfo"); update("MotionEngine", "HeadMotionInfo");
    const auto& joints = rep<JointRequest>("JointRequest"); const auto& info = rep<MotionInfo>("MotionInfo");
    const auto& odometry = rep<OdometryData>("OdometryData");
    json targets=json::array(), stiffness=json::array();
    FOREACH_ENUM(Joints::Joint, joint)
    {
      float angle = joints.angles[joint]; int strength = joints.stiffnessData.stiffnesses[joint];
      if(angle == JointAngles::off || angle == JointAngles::ignore) angle = measured.angles[joint];
      if(strength == StiffnessData::useDefault) strength = rep<StiffnessSettings>("StiffnessSettings").stiffnesses[joint];
      if(!std::isfinite(angle) || strength < 0 || strength > 100) throw std::runtime_error("Invalid original engine output");
      targets.push_back(angle); stiffness.push_back(strength);
    }
    const bool completed = info.lastKickTimestamp > lastKick; lastKick = info.lastKickTimestamp;
    lastSequence = sequence; lastTime = time;
    debug.clear(); annotations.getOut().clear();
    const auto ballZeroStep = Motion::Transformation::getTransformationToZeroStep(
        rep<TorsoMatrix>("TorsoMatrix"),rep<RobotModel>("RobotModel"),rep<RobotDimensions>("RobotDimensions"),
        motion.odometryData,rep<OdometryDataPreview>("OdometryDataPreview"),
        motion.kickType==KickInfo::forwardFastLeft)*motion.ballEstimate.position;
    json output = { {"type","OUTPUT"},{"trial",trial},{"sequence",sequence},{"angles",targets},{"stiffness",stiffness},
       {"phase",TypeRegistry::getEnumName(info.executedPhase)},{"motion",action},{"busy",info.isKicking() || info.executedPhase == MotionPhase::keyframeMotion || info.executedPhase == MotionPhase::getUp},
       {"kick_completed",completed},{"odometry",{odometry.translation.x(),odometry.translation.y(),static_cast<float>(odometry.rotation)}},
       {"motion_info",{{"last_kick_timestamp",info.lastKickTimestamp},{"keyframe",TypeRegistry::getEnumName(info.executedKeyframeMotion.keyframeMotion)},
                       {"mirror",info.executedKeyframeMotion.mirror},{"stable",info.isMotionStable}}},
       {"sensor_info",{{"fall_state",TypeRegistry::getEnumName(rep<FallDownState>("FallDownState").state)},
                       {"ball_zero_step_mm",{ballZeroStep.x(),ballZeroStep.y()}},
                       {"ball_radius_mm",rep<BallSpecification>("BallSpecification").radius},
                       {"inertial_angle",{static_cast<float>(inertial.angle.x()),static_cast<float>(inertial.angle.y()),static_cast<float>(inertial.angle.z())}},
                       {"ground_contact",rep<GroundContactState>("GroundContactState").contact},
                       {"support",rep<FootSupport>("FootSupport").support}}} };
    if(!keeperMotionSummary.is_null())
    {
      output["keeper_motion_skill"] = keeperMotionSummary;
      // Full immutable endpoint details are emitted at application; subsequent
      // 12 ms frames carry the compact confirmation to keep IPC bounded.
      keeperMotionSummary.erase("blocks");
    }
    return output;
  }
};

int main()
{
  std::ios::sync_with_stdio(false);
  FunctionList::execute();
  // KickEngine's original loader locates Config relative to the executable.
  std::filesystem::current_path(BHUMAN_BASELINE_DIR);
  json names=json::array(); FOREACH_ENUM(Joints::Joint,joint) names.push_back(TypeRegistry::getEnumName(joint));
  std::cout << json({{"type","HELLO"},{"protocol",1},{"source_commit",revision},{"cycle_s",.012},
                    {"joint_names",names},{"motions",{"stand","walk","kick","block","getup"}},
                    {"capabilities",{"keeper_defenses_v2","keeper_motion_skill_v1"}},
                    {"keeper_motion_skill",{{"version","keeper_motion_skill_v1"},
                      {"bounds",{{"guard_time_scale",{.85,1.15}},{"jump_time_scale",{.85,1.15}},
                                 {"arm_spread_rad",{-.08,.12}},{"guard_knee_offset_rad",{0.,.10}}}},
                      {"units",{{"guard_time_scale","duration multiplier"},{"jump_time_scale","duration multiplier"},
                                {"arm_spread_rad","rad"},{"guard_knee_offset_rad","rad"}}}}},
                    {"keeper_defenses",{"center","center_right","wide_left","wide_right","left","right"}},
                    {"keeper_preparations",{"prepare"}},
                    {"provenance",json::parse(provenance_json)}}).dump() << std::endl;
  std::unique_ptr<Runtime> runtime;
  std::string line;
  while(true)
  {
    line.clear();
    char character;
    while(std::cin.get(character) && character != '\n')
    {
      if(line.size() >= 65536)
      { std::cout << json({{"type","ERROR"},{"code","MESSAGE_SIZE"},{"message","Message exceeds 65536 bytes"}}).dump()<<std::endl; return 2; }
      line.push_back(character);
    }
    if(!std::cin && line.empty()) break;
    if(!std::cin) { std::cout<<json({{"type","ERROR"},{"code","TRUNCATED_MESSAGE"},{"message","A JSONL message must end with newline"}}).dump()<<std::endl; return 2; }
    try
    {
      auto message=json::parse(line); const std::string type=message.at("type");
      if(type=="SHUTDOWN") break;
      if(type=="INIT")
      {
        if(runtime) throw std::runtime_error("INIT already accepted; start a new process for a reset");
        runtime=std::make_unique<Runtime>(message);
        std::cout<<json({{"type","READY"},{"trial",runtime->trial},{"role",runtime->role},{"protocol",1},
          {"ball_radius_mm",message.value("ball_radius_mm",32.5)}}).dump()<<std::endl;
      }
      else if(type=="STEP")
      {
        if(!runtime) throw std::runtime_error("INIT required before STEP");
        std::cout<<runtime->step(message).dump()<<std::endl;
      }
      else throw std::runtime_error("Unknown message type");
    }
    catch(const std::exception& error)
    { std::cout<<json({{"type","ERROR"},{"code","INVALID_FRAME"},{"message",error.what()}}).dump()<<std::endl; return 2; }
  }
  return 0;
}
