/**
 * @file SkillBehaviorControl.cpp
 *
 * This file implements the behavior control for the skill layer.
 *
 * @author Arne Hasselbring
 */

#include "SkillBehaviorControl.h"
#include "Python/Controller/RLSharedState.h"
#include "Modules/Infrastructure/CameraStreamer/CopaTwinShared.h"
#include "Modules/Perception/BallPerceptors/CopaBallRange.h"
#include "Modules/Perception/CopaPerceptors/CopaVisionShared.h"
#include "Streaming/InStreams.h"
#include "Streaming/TypeRegistry.h"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

MAKE_MODULE(SkillBehaviorControl, SkillBehaviorControl::getExtModuleInfo);

namespace
{
  struct ObstacleSummary
  {
    int count = 0;
    int teammateCount = 0;
    int opponentCount = 0;
    int uncertainCount = 0;
    int frontCount = 0;
    float nearestTeammate = std::numeric_limits<float>::max();
    float nearestOpponent = std::numeric_limits<float>::max();
    float nearestUncertain = std::numeric_limits<float>::max();
    float nearestTeammateFront = std::numeric_limits<float>::max();
    float nearestOpponentFront = std::numeric_limits<float>::max();
    float nearestUncertainFront = std::numeric_limits<float>::max();
    float nearestObstacle = std::numeric_limits<float>::max();
    float nearestObstacleAngle = 0.f;
    float nearestFrontObstacle = std::numeric_limits<float>::max();
    float nearestOpponentAngle = 0.f;
  };

  enum class RLObsExportMode
  {
    estimated = 0,
    corrected = 1,
  };

  RLObsExportMode getRLObsExportMode()
  {
    const char* value = std::getenv("PYBH_RL_OBS_MODE");
    if(!value || !value[0])
      return RLObsExportMode::corrected;
    const std::string mode(value);
    return mode == "estimated" ? RLObsExportMode::estimated : RLObsExportMode::corrected;
  }

  void updateNearest(float distance, bool isFront, float& nearest, float& nearestFront)
  {
    if(distance < nearest)
      nearest = distance;
    if(isFront && distance < nearestFront)
      nearestFront = distance;
  }

  ObstacleSummary summarizeObstacles(const ObstacleModel& obstacleModel)
  {
    ObstacleSummary summary;
    for(const Obstacle& obstacle : obstacleModel.obstacles)
    {
      ++summary.count;
      const float distance = obstacle.center.norm();
      const bool isFront = obstacle.center.x() > 0.f;
      const float angle = std::atan2(obstacle.center.y(), obstacle.center.x());
      if(distance < summary.nearestObstacle)
      {
        summary.nearestObstacle = distance;
        summary.nearestObstacleAngle = angle;
      }
      if(isFront)
      {
        ++summary.frontCount;
        if(distance < summary.nearestFrontObstacle)
          summary.nearestFrontObstacle = distance;
      }
      if(obstacle.isTeammate())
      {
        ++summary.teammateCount;
        updateNearest(distance, isFront, summary.nearestTeammate, summary.nearestTeammateFront);
      }
      else if(obstacle.isOpponent())
      {
        ++summary.opponentCount;
        if(distance < summary.nearestOpponent)
          summary.nearestOpponentAngle = angle;
        updateNearest(distance, isFront, summary.nearestOpponent, summary.nearestOpponentFront);
      }
      else
      {
        ++summary.uncertainCount;
        updateNearest(distance, isFront, summary.nearestUncertain, summary.nearestUncertainFront);
      }
    }
    return summary;
  }

  float boundedDistance(float value, const FieldDimensions& fieldDimensions)
  {
    if(value == std::numeric_limits<float>::max())
      return fieldDimensions.xPosOpponentGroundLine * 2.f;
    return value;
  }
}

SkillBehaviorControl::SkillBehaviorControl() :
  Cabsl<SkillBehaviorControl>(const_cast<ActivationGraph*>(&theActivationGraph)),
  theSkillRegistry("skills.cfg", const_cast<ActivationGraph&>(theActivationGraph), theArmMotionRequest, theBehaviorStatus, theCalibrationRequest, theHeadMotionRequest, theMotionRequest, theOptionalImageRequest)
{}

std::vector<ModuleBase::Info> SkillBehaviorControl::getExtModuleInfo()
{
  auto result = SkillBehaviorControl::getModuleInfo();
  SkillImplementationCreatorBase::addToModuleInfo(SkillImplementationCreatorList<Skill>::first, result);
  return result;
}

void SkillBehaviorControl::update(ActivationGraph&)
{
  DECLARE_DEBUG_RESPONSE("option:HandleRefereeSignal:now");

  if(theMotionInfo.lastKickTimestamp > lastReportedKickTimestamp)
  {
    lastReportedKickTimestamp = theMotionInfo.lastKickTimestamp;
    const float centerCircleRadius = theFieldDimensions.centerCircleRadius +
                                     theFieldDimensions.fieldLinesWidth * 0.5f +
                                     theBallSpecification.radius;
    lastReportedKickWasOutsideCenterCircle = theFieldBall.positionOnField.squaredNorm() >= sqr(centerCircleRadius);
  }

  theBehaviorStatus.calibrationFinished = false;
  theBehaviorStatus.lastKickTimestamp = lastReportedKickTimestamp;
  theBehaviorStatus.lastKickWasOutsideCenterCircle = lastReportedKickWasOutsideCenterCircle;
  theBehaviorStatus.passTarget = -1;
  theBehaviorStatus.passOrigin = -1;
  theBehaviorStatus.walkingTo = Vector2f::Zero();
  theBehaviorStatus.speed = 0.f;
  theBehaviorStatus.shootingTo.reset();
  theBehaviorStatus.restartMemoryType = theRestartBallSearchContext.restartType;
  theBehaviorStatus.restartMemoryRegionIndex = theRestartBallSearchContext.regionIndex;
  theBehaviorStatus.restartMemoryPositionOnField = theRestartBallSearchContext.rememberedPositionOnField;
  theBehaviorStatus.restartMemoryTimestamp = theRestartBallSearchContext.sourceTimestamp;
  theBehaviorStatus.restartMemorySourceRobot = theRestartBallSearchContext.sourceRobotNumber;
  theBehaviorStatus.restartMemoryValid = theRestartBallSearchContext.valid;
  theBehaviorStatus.restartMemoryFrozen = theRestartBallSearchContext.frozenForCurrentRestart;
  theBehaviorStatus.restartMemoryFromLiveBall = theRestartBallSearchContext.fromLiveBall;
  theBehaviorStatus.restartMemoryFromDropInFallback = theRestartBallSearchContext.fromDropInFallback;

  theArmMotionRequest.armMotion[Arms::left] = ArmMotionRequest::none;
  theArmMotionRequest.armMotion[Arms::right] = ArmMotionRequest::none;

  theHeadMotionRequest.mode = HeadMotionRequest::panTiltMode;
  theHeadMotionRequest.cameraControlMode = HeadMotionRequest::autoCamera;
  theHeadMotionRequest.pan = JointAngles::off;
  theHeadMotionRequest.tilt = JointAngles::off;
  theHeadMotionRequest.speed = 150_deg;
  theHeadMotionRequest.stopAndGoMode = false;

  theMotionRequest.shouldInterceptBall = false;
  theMotionRequest.odometryData = theOdometryData;
  theMotionRequest.ballEstimate = theBallModel.estimate;
  theMotionRequest.ballEstimateTimestamp = theFrameInfo.time;
  theMotionRequest.ballTimeWhenLastSeen = theBallModel.timeWhenLastSeen;
  theMotionRequest.shouldInterceptBall = useNewHandleCatchBallBehavior && theFieldBall.interceptBall;

  theOptionalImageRequest.sendImage = false;

  // On request, tell the user whether a USB drive is mounted.
  if(theEnhancedKeyStates.hitStreak[KeyStates::headMiddle] == 3)
  {
    if(SystemCall::usbIsMounted())
      SystemCall::say("USB mounted");
    else
      SystemCall::say("USB not mounted");
  }

  // On request, tell the user the temperature of the hottest joint.
  if((theEnhancedKeyStates.hitStreak[KeyStates::headRear] == 3 && theEnhancedKeyStates.pressedDuration[KeyStates::headFront] > 0) ||
     (theEnhancedKeyStates.hitStreak[KeyStates::headFront] == 3 && theEnhancedKeyStates.pressedDuration[KeyStates::headRear] > 0))
    SystemCall::say((std::string(TypeRegistry::getEnumName(theGameState.color())) + " " + std::to_string(theGameState.playerNumber) + " " +
                     std::regex_replace(TypeRegistry::getEnumName(theRobotHealth.jointWithMaxTemperature), std::regex("([A-Z])"), " $1") + " " +
                     std::to_string(theJointSensorData.temperatures[theRobotHealth.jointWithMaxTemperature])).c_str(), true);
  else if(theEnhancedKeyStates.hitStreak[KeyStates::headFront] == 3)
  {
    // On request, tell the user the deployed configuration of this robot.
    std::string wavName = Global::getSettings().headName.c_str();
    wavName.append(".wav");
    SystemCall::playSound(wavName.c_str());
    SystemCall::say(("  my name is  " + theRobotHealth.robotName+ " im wearing  " +
                     std::string(TypeRegistry::getEnumName(theGameState.color())) +
                     " with the number " +
                     std::to_string(theGameState.playerNumber)).c_str(), true, 0.7f);
  }

  theSkillRegistry.modifyAllParameters();

  theSkillRegistry.preProcess(theFrameInfo.time);
  beginFrame(theFrameInfo.time);

  if(RLSharedStateBridge::isEnabledForTeam(theGameState.ownTeam.number))
  {
    const int n = theGameState.playerNumber > 0 ? theGameState.playerNumber : 1;
    RLPlayerIO& io = RLSharedState::instance().player(n);
    io.lock();
    io.debugZweikampfActive = false;
    io.unlock();
  }

  const std::string& scenario = Global::getSettings().scenario;
  if(scenario == "CopaPenaltyStriker" || scenario == "CopaPenaltyKeeper")
    executeCopaPenalty(scenario == "CopaPenaltyKeeper");
  else
    PlaySoccer();

  const auto ownRestartName = [this]() -> const char*
  {
    if(theGameState.isKickIn())
      return "kick in";
    if(theGameState.isGoalKick())
      return "goal kick";
    if(theGameState.isCornerKick())
      return "corner kick";
    if(theGameState.state == GameState::ownDirectFreeKick || theGameState.state == GameState::ownPushingFreeKick)
      return "direct free kick";
    if(theGameState.state == GameState::ownIndirectFreeKick)
      return "indirect free kick";
    if(theGameState.isPenaltyKick())
      return "penalty kick";
    if(theGameState.isKickOff())
      return "kick off";
    return "restart";
  };

  const bool isOwnRestart = theGameState.isForOwnTeam() &&
                            (theGameState.isKickOff() || theGameState.isPenaltyKick() || theGameState.isFreeKick());
  const bool isSetPlayExecutor = isOwnRestart &&
                                 theStrategyStatus.acceptedSetPlay != SetPlay::none &&
                                 theStrategyStatus.setPlayStep >= 0 &&
                                 (theStrategyStatus.role == ActiveRole::toRole(ActiveRole::startSetPlay) ||
                                  theStrategyStatus.role == ActiveRole::toRole(ActiveRole::playBall));

  if(isSetPlayExecutor &&
     (lastAnnouncedOwnRestartStateStarted != theGameState.timeWhenStateStarted ||
      lastAnnouncedOwnRestartSetPlay != theStrategyStatus.acceptedSetPlay ||
      lastAnnouncedOwnRestartPlayer != theGameState.playerNumber))
  {
    const std::string announcement = "I am executing " + std::string(ownRestartName()) + " player " +
                                     std::to_string(theGameState.playerNumber);
    SystemCall::say(announcement.c_str());
    lastAnnouncedOwnRestartStateStarted = theGameState.timeWhenStateStarted;
    lastAnnouncedOwnRestartSetPlay = theStrategyStatus.acceptedSetPlay;
    lastAnnouncedOwnRestartPlayer = theGameState.playerNumber;
  }

  endFrame();
  theSkillRegistry.postProcess();

  if(RLSharedStateBridge::isEnabledForTeam(theGameState.ownTeam.number))
  {
    const int n = theGameState.playerNumber > 0 ? theGameState.playerNumber : 1;
    RLPlayerIO& io = RLSharedState::instance().player(n);
    bool postObs = false;
    io.lock();
    io.debugSkillBehaviorSkillRequest = static_cast<int>(theSkillRequest.skill);
    io.debugSkillBehaviorMotionRequest = static_cast<int>(theMotionRequest.motion);
    ++io.debugSkillBehaviorCallCount;
    io.debugSkillBehaviorWalkTargetX = theMotionRequest.walkTarget.translation.x();
    io.debugSkillBehaviorWalkTargetY = theMotionRequest.walkTarget.translation.y();
    io.debugSkillBehaviorWalkTargetTheta = static_cast<float>(theMotionRequest.walkTarget.rotation);
    io.debugMotionObstacleAvoidanceX = theMotionRequest.obstacleAvoidance.avoidance.x();
    io.debugMotionObstacleAvoidanceY = theMotionRequest.obstacleAvoidance.avoidance.y();
    io.debugMotionObstaclePathCount = static_cast<int>(theMotionRequest.obstacleAvoidance.path.size());
    if(!theMotionRequest.obstacleAvoidance.path.empty())
    {
      const auto& firstSegment = theMotionRequest.obstacleAvoidance.path.front();
      io.debugMotionObstacleFirstX = firstSegment.obstacle.center.x();
      io.debugMotionObstacleFirstY = firstSegment.obstacle.center.y();
      io.debugMotionObstacleFirstRadius = firstSegment.obstacle.radius;
      io.debugMotionObstacleFirstClockwise = firstSegment.clockwise;
    }
    else
    {
      io.debugMotionObstacleFirstX = 0.f;
      io.debugMotionObstacleFirstY = 0.f;
      io.debugMotionObstacleFirstRadius = 0.f;
      io.debugMotionObstacleFirstClockwise = false;
    }
    const Vector2f ballOnField = theFieldBall.positionOnField;
    const Vector2f ballRelative = theFieldBall.positionRelative;
    const Vector2f ballEndOnField = theFieldBall.endPositionOnField;
    const Vector2f ballEndRelative = theFieldBall.endPositionRelative;
    const Vector2f ballVelocity = theBallModel.estimate.velocity;
    const ObstacleSummary obstacleSummary = summarizeObstacles(theObstacleModel);

    const RLObsExportMode obsExportMode = getRLObsExportMode();
    const float estimatedRobotX = theRobotPose.translation.x();
    const float estimatedRobotY = theRobotPose.translation.y();
    const float estimatedRobotTheta = static_cast<float>(theRobotPose.rotation);
    const float estimatedBallX = ballOnField.x();
    const float estimatedBallY = ballOnField.y();
    float exportedRobotX = estimatedRobotX;
    float exportedRobotY = estimatedRobotY;
    float exportedRobotTheta = estimatedRobotTheta;
    float exportedBallX = estimatedBallX;
    float exportedBallY = estimatedBallY;
    Vector2f exportedBallRelative = ballRelative;
    Vector2f exportedBallEndRelative = ballEndRelative;
    float exportedTimeSinceBallSeen = static_cast<float>(theFieldBall.timeSinceBallWasSeen);
    float exportedTimeSinceBallDisappeared = static_cast<float>(theFieldBall.timeSinceBallDisappeared);
    float exportedBallSeenPercentage = static_cast<float>(theBallModel.seenPercentage);
    bool exportedBallConsistent = theFieldBall.ballPositionConsistentWithGameState;
    const float naturalTimeSinceBallSeen = exportedTimeSinceBallSeen;
    const float naturalBallSeenPercentage = exportedBallSeenPercentage;
    const bool naturalBallConsistent = exportedBallConsistent;
    bool correctedRobotPose = false;
    bool correctedBall = false;
    int ballExportSource = 0; // 0=estimated, 1=reset fallback, 2=dynamic fallback

    const float simPoseError = std::hypot(estimatedRobotX - io.simRobotX, estimatedRobotY - io.simRobotY);
    if(obsExportMode == RLObsExportMode::corrected && simPoseError > 500.f)
    {
      exportedRobotX = io.simRobotX;
      exportedRobotY = io.simRobotY;
      exportedRobotTheta = io.simRobotTheta;
      correctedRobotPose = true;
    }

    const bool staleBall = theBallModel.seenPercentage <= 0.f && theFieldBall.timeSinceBallWasSeen > 1000;
    if(obsExportMode == RLObsExportMode::corrected && staleBall)
    {
      exportedBallX = io.dynamicHasBall ? io.dynamicBallX : io.resetBallX;
      exportedBallY = io.dynamicHasBall ? io.dynamicBallY : io.resetBallY;
      exportedTimeSinceBallSeen = 0.f;
      exportedTimeSinceBallDisappeared = 0.f;
      exportedBallSeenPercentage = 100.f;
      exportedBallConsistent = true;
      correctedBall = true;
      ballExportSource = io.dynamicHasBall ? 2 : 1;
    }

    const Pose2f exportedRobotPose(exportedRobotTheta, exportedRobotX, exportedRobotY);
    if(correctedRobotPose || correctedBall)
    {
      const Vector2f exportedBallOnField(exportedBallX, exportedBallY);
      exportedBallRelative = exportedRobotPose.inverse() * exportedBallOnField;
      exportedBallEndRelative = correctedBall
                                   ? exportedBallRelative
                                   : exportedRobotPose.inverse() * ballEndOnField;
    }

    const Vector2f exportedBallOnField(exportedBallX, exportedBallY);
    const float exportedShotQualityNoObstacles = theExpectedGoals.xG ? theExpectedGoals.xG(exportedBallOnField) : 0.f;
    const float exportedShotOpeningWithObstacles = theExpectedGoals.getRating ? theExpectedGoals.getRating(exportedBallOnField) : 0.f;

    io.ballX = exportedBallX;
    io.ballY = exportedBallY;
    io.robotX = exportedRobotX;
    io.robotY = exportedRobotY;
    io.robotTheta = exportedRobotTheta;
    io.estimatedBallX = estimatedBallX;
    io.estimatedBallY = estimatedBallY;
    io.estimatedRobotX = estimatedRobotX;
    io.estimatedRobotY = estimatedRobotY;
    io.estimatedRobotTheta = estimatedRobotTheta;
    io.ballRelX = exportedBallRelative.x();
    io.ballRelY = exportedBallRelative.y();
    io.ballEndRelX = exportedBallEndRelative.x();
    io.ballEndRelY = exportedBallEndRelative.y();
    io.obsExportMode = static_cast<int>(obsExportMode);
    io.correctedExportedRobotPose = correctedRobotPose;
    io.correctedExportedBall = correctedBall;
    io.ballExportSource = ballExportSource;
    io.ballExportFresh = !correctedBall;
    io.naturalTimeSinceBallSeen = naturalTimeSinceBallSeen;
    io.naturalBallSeenPercentage = naturalBallSeenPercentage;
    io.naturalBallConsistentWithGameState = naturalBallConsistent;
    io.ballVelX = ballVelocity.x();
    io.ballVelY = ballVelocity.y();
    io.timeSinceBallSeen = exportedTimeSinceBallSeen;
    io.timeSinceBallDisappeared = exportedTimeSinceBallDisappeared;
    io.ballSeenPercentage = exportedBallSeenPercentage;
    io.ballConsistentWithGameState = exportedBallConsistent;
    io.canScoreNow = exportedShotOpeningWithObstacles > 0.8f;
    io.shotQualityNoObstacles = exportedShotQualityNoObstacles;
    io.shotOpeningWithObstacles = exportedShotOpeningWithObstacles;
    io.passOptionsCount = static_cast<float>(theTeamData.teammates.size());
    io.nearestTeammateDist = boundedDistance(obstacleSummary.nearestTeammate, theFieldDimensions);
    io.nearestOpponentDist = boundedDistance(obstacleSummary.nearestOpponent, theFieldDimensions);
    io.nearestUncertainObstacleDist = boundedDistance(obstacleSummary.nearestUncertain, theFieldDimensions);
    io.nearestTeammateFrontDist = boundedDistance(obstacleSummary.nearestTeammateFront, theFieldDimensions);
    io.nearestOpponentFrontDist = boundedDistance(obstacleSummary.nearestOpponentFront, theFieldDimensions);
    io.nearestUncertainFrontDist = boundedDistance(obstacleSummary.nearestUncertainFront, theFieldDimensions);
    io.debugObstacleCount = obstacleSummary.count;
    io.debugObstacleTeammateCount = obstacleSummary.teammateCount;
    io.debugObstacleOpponentCount = obstacleSummary.opponentCount;
    io.debugObstacleUncertainCount = obstacleSummary.uncertainCount;
    io.debugObstacleFrontCount = obstacleSummary.frontCount;
    io.debugNearestObstacleDist = boundedDistance(obstacleSummary.nearestObstacle, theFieldDimensions);
    io.debugNearestObstacleAngle = obstacleSummary.nearestObstacleAngle;
    io.debugNearestFrontObstacleDist = boundedDistance(obstacleSummary.nearestFrontObstacle, theFieldDimensions);
    io.debugNearestOpponentAngle = obstacleSummary.nearestOpponentAngle;
    io.frame = theFrameInfo.time;
    io.obsReady = true;
    postObs = true;
    io.unlock();
    if(postObs)
      io.postObs();
  }

  theLibCheck.performCheck();
}

void SkillBehaviorControl::executeCopaPenalty(bool keeper)
{
  if(!keeper && !copaShotConfigLoaded)
  {
    InMapFile stream("copaShot.cfg");
    if(stream.exists())
      stream >> copaShotConfig;
    copaShotConfigLoaded = true;
  }
  if(keeper && !copaKeeperConfigLoaded)
  {
    InMapFile stream("copaKeeper.cfg");
    if(stream.exists())
      stream >> copaKeeperConfig;
    copaKeeperConfigLoaded = true;
  }
  if(theCameraInfo.camera == CameraInfo::lower)
    copaLowerCamera = theCameraMatrix.translation;
  if(!copaInitialized)
  {
    resetCopaRun();
    copaInitialized = true;
  }
  const bool threeHeads = theEnhancedKeyStates.pressed[KeyStates::headFront] &&
                          theEnhancedKeyStates.pressed[KeyStates::headMiddle] &&
                          theEnhancedKeyStates.pressed[KeyStates::headRear];
  const bool unstiff = theGameState.playerState == GameState::unstiff;
  const bool shutdownLatched = copaShutdown.update(threeHeads, unstiff);
  const bool startTouch = copaHeadGate.start(theEnhancedKeyStates.pressed[KeyStates::headFront],
                                             theEnhancedKeyStates.pressed[KeyStates::headMiddle],
                                             theEnhancedKeyStates.pressed[KeyStates::headRear]);
  const auto state = copaState.update(theFrameInfo.time, theWhistle.lastTimeWhistleDetected,
                                      startTouch,
                                      theEnhancedKeyStates.pressed[KeyStates::chest],
                                      theEnhancedKeyStates.pressed[KeyStates::headRear] &&
                                        !theEnhancedKeyStates.pressed[KeyStates::headFront] &&
                                        !theEnhancedKeyStates.pressed[KeyStates::headMiddle], shutdownLatched || unstiff);
  publishCopaTwin(keeper, static_cast<int>(state));
  copaWalkValid = false; // set again by this frame's walk request

  if(state != previousCopaState)
  {
    OUTPUT_TEXT("CopaPenalty state=" << static_cast<int>(state) << " role=" << (keeper ? "keeper" : "striker"));
    const char* role = keeper ? "goalkeeper" : "striker";
    const char* name = state == CopaPenaltyState::waiting ? "armed" : state == CopaPenaltyState::running ? "start" : "stop";
    SystemCall::say((std::string(role) + " mode " + name).c_str());
    if(state == CopaPenaltyState::running)
    {
      // The 28 s limit runs from the whistle, not from when it was recognized.
      OUTPUT_WARNING("[CopaStart] trigger=" << (copaState.startedByWhistle() ? "whistle" : "headFront")
                     << " role=" << role << " whistleConfidence=" << theWhistle.confidenceOfLastWhistleDetection
                     << " delayMs=" << static_cast<int>(theFrameInfo.time - copaState.shotStart()));
      copaKeeper.reset();
      copaKeeperStart = theOdometryData;
      copaKeeperLastAction = CopaKeeper::Action::watch;
      copaBallTrack.reset();
      // Striker mode keeps the twin built when armed; it re-takes the mark and scans again.
      copaPhase = CopaStrikerPhase::findBall;
      copaPhaseStart = theFrameInfo.time;
      copaSeen = CopaShot::Observations();
      copaShotKnown = copaPreKickValid = copaWalkValid = false;
      copaMinFootClearance = std::numeric_limits<float>::max();
    }
    if(state == CopaPenaltyState::waiting)
    {
      // Armed: a new run and a new twin, anchored on the ball in front of the feet.
      ++copaRun;
      resetCopaRun();
      copaPhase = CopaStrikerPhase::scanBall;
    }
    if(state == CopaPenaltyState::stopped)
      resetCopaRun(); // cleared until the next arm
    previousCopaState = state;
  }

  if(shutdownLatched || unstiff)
  {
    if(threeHeads && !copaShutdownAnnounced)
    {
      SystemCall::say("sitting down");
      copaShutdownAnnounced = true;
    }
    if(!threeHeads)
      copaShutdownAnnounced = false;
    theLookAtAnglesSkill({.pan = JointAngles::off, .tilt = JointAngles::off});
    thePlayDeadSkill();
    return;
  }
  copaShutdownAnnounced = false;

  const bool ballFresh = theBallModel.timeWhenLastSeen != 0 && theFrameInfo.getTimeSince(theBallModel.timeWhenLastSeen) < 500;
  if(!keeper && theCameraStatus.ok && state == CopaPenaltyState::waiting)
  {
    executeCopaArmedScan(ballFresh);
    return;
  }
  if(!theCameraStatus.ok || state != CopaPenaltyState::running)
  {
    if(!theCameraStatus.ok)
      copaState.finish();
    theLookForwardSkill();
    theStandSkill({.high = true});
    return;
  }

  if(keeper)
    executeCopaKeeper(ballFresh);
  else if(theMotionInfo.lastKickTimestamp != copaLastKick)
  {
    copaState.finish();
    theLookForwardSkill();
    theStandSkill({.high = true});
  }
  else
    executeCopaStriker(ballFresh);
}

void SkillBehaviorControl::resetCopaRun()
{
  const CopaShotConfig& c = copaShotConfig;
  copaPhase = CopaStrikerPhase::ready;
  copaPhaseStart = theFrameInfo.time;
  copaStartOdometry = theOdometryData;
  // Until the ball is seen: the mark startBallFromToes in front of the toes, facing the goal.
  copaBallOdometry = theOdometryData * Vector2f(theRobotDimensions.soleToFrontEdgeLength + c.startBallFromToes, 0.f);
  copaMarkKnown = true;
  copaMarkSeen = false;
  copaHeadingCorrection = copaArmedCorrection = 0.f;
  copaPostsMatched = 0;
  rebuildCopaGoalFromMark(0.f);
  copaShotKnown = copaPreKickValid = copaWalkValid = false;
  copaShot = CopaShot::Shot();
  copaSeen = CopaShot::Observations();
  copaLastSnapshot = 0;
  copaFootClearance = 0.f;
  copaMinFootClearance = std::numeric_limits<float>::max();
  copaTrackShift = Vector2f::Zero();
  copaLastBallSeen = 0;
  copaKickType = KickInfo::forwardFastLeftLong;
}

void SkillBehaviorControl::rebuildCopaGoalFromMark(float correction)
{
  const CopaShotConfig& c = copaShotConfig;
  const Vector2f forward = Vector2f(1.f, 0.f).rotated(copaStartOdometry.rotation + correction);
  const Vector2f center = copaBallOdometry + forward * c.penaltyDistance;
  const Vector2f side = forward.rotated(90_deg) * (0.5f * c.goalWidth);
  copaLeftPostOdometry = center + side;
  copaRightPostOdometry = center - side;
}

bool SkillBehaviorControl::captureCopaMark(bool calibrate)
{
  const CopaShotConfig& c = copaShotConfig;
  Vector2f ball = theBallModel.estimate.position;
  if(ball.x() <= 0.f || ball.norm() > c.maxMarkDistance)
    return false;
  copaRangeMeasured = ball.norm();
  if(calibrate)
  {
    // Armed: the ball is on the mark startBallFromToes in front of the toes. Refine the lower
    // camera's pitch offset so the measured range matches; the detector applies it from now on.
    const float trueX = theRobotDimensions.soleToFrontEdgeLength + c.startBallFromToes;
    const float step = CopaRange::pitchOffsetFromKnown(copaLowerCamera.x(), copaLowerCamera.y(), copaLowerCamera.z(),
                                                       theBallSpecification.radius, ball.x(), ball.y(), trueX);
    const float before = CopaRange::lowerPitchOffset().load();
    const float total = std::clamp(before + step, -static_cast<float>(c.maxPitchCorrection), static_cast<float>(c.maxPitchCorrection));
    CopaRange::lowerPitchOffset().store(total);
    float x = ball.x(), y = ball.y();
    CopaRange::correctRange(copaLowerCamera.x(), copaLowerCamera.y(), copaLowerCamera.z(), theBallSpecification.radius,
                            total - before, x, y);
    OUTPUT_WARNING("[CopaShot] range calibration: measured=(" << static_cast<int>(ball.x()) << "," << static_cast<int>(ball.y())
                   << ") known x=" << static_cast<int>(trueX) << " corrected=(" << static_cast<int>(x) << "," << static_cast<int>(y)
                   << ") pitchOffset=" << Angle(total).toDegrees() << "deg cameraHeight=" << static_cast<int>(copaLowerCamera.z()));
    ball = Vector2f(x, y);
  }
  copaBallOdometry = theOdometryData * ball;
  copaMarkSeen = true;
  rebuildCopaGoalFromMark(copaHeadingCorrection);
  OUTPUT_WARNING("[CopaShot] mark = ball seen at (" << static_cast<int>(ball.x()) << "," << static_cast<int>(ball.y()) << ")");
  return true;
}

CopaShot::Feet SkillBehaviorControl::copaFeet() const
{
  CopaShot::Feet feet;
  feet.hip = theRobotDimensions.yHipOffset;
  feet.front = theRobotDimensions.soleToFrontEdgeLength;
  feet.back = theRobotDimensions.soleToBackEdgeLength;
  feet.inner = theRobotDimensions.soleToInnerEdgeLength;
  feet.outer = theRobotDimensions.soleToOuterEdgeLength;
  return feet;
}

void SkillBehaviorControl::trackCopaMark(bool ballFresh)
{
  const CopaShotConfig& c = copaShotConfig;
  if(!ballFresh || theBallModel.timeWhenLastSeen == copaLastBallSeen)
    return;
  copaLastBallSeen = theBallModel.timeWhenLastSeen;
  const Vector2f predicted = theOdometryData.inverse() * copaBallOdometry;
  const Vector2f error = theBallModel.estimate.position - predicted;
  if(error.norm() > c.trackGate)
    return;
  // The ball rests on the mark: a difference is odometry drift. Move the whole field with it.
  const Vector2f delta = error.rotated(theOdometryData.rotation) * c.trackGain;
  copaBallOdometry += delta;
  copaTargetOdometry += delta;
  copaLeftPostOdometry += delta;
  copaRightPostOdometry += delta;
  copaKeeperOdometry += delta;
  copaTrackShift += delta;
}

Pose2f SkillBehaviorControl::clampCopaTarget(const Pose2f& relative) const
{
  const CopaShotConfig& c = copaShotConfig;
  // Ball frame: origin on the mark, x towards the goal (placement heading + correction).
  const Pose2f ballFrame(copaStartOdometry.rotation + copaHeadingCorrection, copaBallOdometry);
  const Pose2f inBall = ballFrame.inverse() * (theOdometryData * relative);
  CopaShot::KickBox box;
  box.maxBack = c.boxMaxBack;
  box.maxLateral = c.boxMaxLateral;
  box.lineMargin = c.minFootClearance;
  box.maxTurn = c.maxAim;
  const CopaShot::Feet feet = copaFeet();
  CopaShot::Pose pose;
  pose.rotation = inBall.rotation;
  pose.x = inBall.translation.x();
  pose.y = inBall.translation.y();
  pose = CopaShot::clampToKickBox(pose, box, feet.front, feet.hip, feet.outer, theBallSpecification.radius);
  return theOdometryData.inverse() * (ballFrame * Pose2f(pose.rotation, pose.x, pose.y));
}

bool SkillBehaviorControl::sweepCopaGoal()
{
  const CopaShotConfig& c = copaShotConfig;
  const Angle pans[3] = {c.sweepPan, 0_deg, -c.sweepPan};
  const int elapsed = static_cast<int>(theFrameInfo.getTimeSince(copaPhaseStart));
  const int step = elapsed / std::max(1, c.sweepStepMs);
  if(step >= 3)
    return true;
  theLookAtAnglesSkill({.pan = pans[step], .tilt = c.headTilt, .speed = 150_deg});
  const CopaVisionSnapshot snapshot = CopaVisionShared::instance().read();
  const unsigned settled = copaPhaseStart + static_cast<unsigned>(step * c.sweepStepMs + c.settleMs);
  if(snapshot.time != copaLastSnapshot && snapshot.time >= settled)
  {
    copaLastSnapshot = snapshot.time;
    copaSeen.posts.insert(copaSeen.posts.end(), snapshot.postBearings.begin(), snapshot.postBearings.end());
    copaSeen.robots.insert(copaSeen.robots.end(), snapshot.robotBearings.begin(), snapshot.robotBearings.end());
  }
  return false;
}

void SkillBehaviorControl::executeCopaArmedScan(bool ballFresh)
{
  const CopaShotConfig& c = copaShotConfig;
  theStandSkill({.high = true});
  auto enter = [this](CopaStrikerPhase phase)
  {
    copaPhase = phase;
    copaPhaseStart = theFrameInfo.time;
  };
  switch(copaPhase)
  {
    case CopaStrikerPhase::scanBall:
      if(ballFresh)
        theLookAtBallSkill();
      else
        theLookAtAnglesSkill({.pan = 0_deg, .tilt = c.ballScanTilt, .speed = 150_deg});
      if(ballFresh && captureCopaMark(true))
        SystemCall::say("ball on the mark");
      else if(theFrameInfo.getTimeSince(copaPhaseStart) < c.ballScanMs)
        return;
      else
      {
        OUTPUT_WARNING("[CopaShot] armed: ball not seen, mark assumed " << static_cast<int>(c.startBallFromToes) << "mm in front of the toes");
        SystemCall::say("ball not seen");
      }
      copaSeen = CopaShot::Observations();
      copaLastSnapshot = 0;
      enter(CopaStrikerPhase::scanGoal);
      return;

    case CopaStrikerPhase::scanGoal:
    {
      if(!sweepCopaGoal())
        return;
      rebuildCopaGoalFromMark(0.f);
      const Pose2f toRobot = theOdometryData.inverse();
      const Vector2f left = toRobot * copaLeftPostOdometry, right = toRobot * copaRightPostOdometry;
      CopaShot::PriorGoal prior;
      prior.leftX = left.x();
      prior.leftY = left.y();
      prior.rightX = right.x();
      prior.rightY = right.y();
      copaArmedCorrection = CopaShot::headingCorrection(copaSeen, prior, c.postMatchTolerance, c.maxHeadingCorrection,
                                                        copaPostsMatched);
      copaHeadingCorrection = copaArmedCorrection;
      rebuildCopaGoalFromMark(copaHeadingCorrection);
      OUTPUT_WARNING("[CopaShot] armed: mark=" << (copaMarkSeen ? "seen" : "assumed") << " postSamples="
                     << static_cast<unsigned>(copaSeen.posts.size()) << " postsMatched=" << copaPostsMatched
                     << " headingCorrection=" << Angle(copaHeadingCorrection).toDegrees() << "deg");
      SystemCall::say(copaPostsMatched > 0 ? "goal confirmed" : "goal not seen");
      enter(CopaStrikerPhase::ready);
      return;
    }

    default:
      if(ballFresh)
        theLookAtBallSkill();
      else
        theLookForwardSkill();
      return;
  }
}

void SkillBehaviorControl::publishCopaTwin(bool keeper, int state)
{
  // Real-time tracking moved the field by copaTrackShift; show it the other way round: the field
  // stays where it was built and the robot's corrected position moves.
  const Pose2f fromOdometry = copaStartOdometry.inverse();
  Pose2f corrected = theOdometryData;
  corrected.translation -= copaTrackShift;
  const Pose2f robot = fromOdometry * corrected;
  auto point = [&](bool valid, const Vector2f& odometry)
  {
    CopaTwinSnapshot::Point p;
    p.valid = valid;
    const Vector2f v = fromOdometry * Vector2f(odometry - copaTrackShift);
    p.x = v.x();
    p.y = v.y();
    return p;
  };
  auto pose = [&](bool valid, const Pose2f& relative)
  {
    CopaTwinSnapshot::Pose p;
    p.valid = valid;
    const Pose2f v = robot * relative;
    p.x = v.translation.x();
    p.y = v.translation.y();
    p.rotation = v.rotation;
    return p;
  };

  CopaTwinSnapshot twin;
  twin.time = theFrameInfo.time;
  twin.run = copaRun;
  twin.role = keeper ? 1 : 0;
  twin.state = static_cast<uint8_t>(state);
  twin.phase = static_cast<uint8_t>(keeper ? static_cast<int>(copaKeeperLastAction) : static_cast<int>(copaPhase));
  twin.robot = pose(true, Pose2f());
  twin.ball = point(theFieldBall.ballWasSeen(1000), theOdometryData * theBallModel.estimate.position);
  twin.mark = point(copaMarkKnown, copaBallOdometry);
  twin.leftPost = point(!keeper, copaLeftPostOdometry);
  twin.rightPost = point(!keeper, copaRightPostOdometry);
  twin.goalInferred = copaPostsMatched == 0; // goal only from the mark and placement, no post confirmed it
  twin.markSeen = copaMarkSeen;
  twin.boxMaxBack = copaShotConfig.boxMaxBack;
  twin.boxMaxLateral = copaShotConfig.boxMaxLateral;
  twin.lineDistance = theBallSpecification.radius + copaShotConfig.minFootClearance;
  twin.goalHeading = copaHeadingCorrection;
  twin.pitchOffset = CopaRange::lowerPitchOffset().load();
  twin.trackShift = copaTrackShift.norm();
  twin.headingCorrection = copaHeadingCorrection;
  twin.postsMatched = static_cast<uint8_t>(std::min(copaPostsMatched, 255));
  twin.keeper = point(copaShotKnown && copaShot.keeperSeen, copaKeeperOdometry);
  twin.target = point(copaShotKnown, copaTargetOdometry);
  twin.side = static_cast<int8_t>(copaShot.side);
  twin.preKick = pose(copaPreKickValid, copaPreKickRelative);
  twin.walkTarget = pose(copaWalkValid, copaWalkTargetRelative);
  twin.footClearance = copaFootClearance;
  twin.minFootClearance = copaMinFootClearance == std::numeric_limits<float>::max() ? 0.f : copaMinFootClearance;
  twin.ballFromMark = copaMarkKnown ? (theOdometryData * theBallModel.estimate.position - copaBallOdometry).norm() : 0.f;
  CopaTwinShared::instance().publish(twin);
}

void SkillBehaviorControl::executeCopaKeeper(bool ballFresh)
{
  const CopaKeeperConfig& c = copaKeeperConfig;
  CopaKeeper::Params params;
  params.minApproachSpeed = c.minApproachSpeed;
  params.maxTime = c.maxTime;
  params.genuflectTime = c.genuflectTime;
  params.genuflectDistance = c.genuflectDistance;
  params.standRadius = c.standRadius;
  params.maxLateral = c.maxLateral;
  params.confirmMs = c.confirmMs;
  params.holdMs = c.holdMs;
  params.releaseMs = c.releaseMs;
  params.walkStartMargin = c.walkStartMargin;

  CopaKeeper::Ball ball;
  ball.seen = theFieldBall.ballWasSeen(c.seenTimeoutMs);
  ball.x = theBallModel.estimate.position.x();
  ball.y = theBallModel.estimate.position.y();
  ball.vx = theBallModel.estimate.velocity.x();
  ball.crossY = theFieldBall.intersectionPositionWithOwnYAxis.y();
  ball.crossTime = theFieldBall.timeUntilIntersectsOwnYAxis;

  const float lateral = (copaKeeperStart.inverse() * theOdometryData).translation.y();

  // Own velocity from the recent sightings, fixed to the floor through odometry, so the
  // keeper's own steps are not ball motion. Preferred over B-Human's estimate, which did
  // not report hand-rolled shots with the 3-4 Hz detections.
  const int now = static_cast<int>(theFrameInfo.time);
  if(theBallModel.timeWhenLastSeen != 0 && theFrameInfo.getTimeSince(theBallModel.timeWhenLastSeen) < 100)
  {
    const Vector2f onFloor = theOdometryData * theBallModel.lastPerception;
    copaBallTrack.add(static_cast<int>(theBallModel.timeWhenLastSeen), onFloor.x(), onFloor.y());
  }
  float fitX = 0.f, fitY = 0.f;
  const bool fitted = ball.seen && copaBallTrack.velocity(now, c.velocityWindowMs, fitX, fitY);
  Vector2f ownVelocity = Vector2f::Zero();
  if(fitted)
  {
    ownVelocity = Vector2f(fitX, fitY).rotated(-theOdometryData.rotation);
    ball.vx = ownVelocity.x();
    float crossY = 0.f, crossTime = 0.f;
    if(CopaKeeper::crossing(ball.x, ball.y, ownVelocity.x(), ownVelocity.y(), crossY, crossTime))
    {
      ball.crossY = crossY;
      ball.crossTime = crossTime;
    }
  }
  if(c.telemetryMs > 0 && ball.seen && theFrameInfo.getTimeSince(copaKeeperTelemetry) >= c.telemetryMs)
  {
    copaKeeperTelemetry = theFrameInfo.time;
    OUTPUT_WARNING("[CopaKeeperBall] ball=(" << static_cast<int>(ball.x) << "," << static_cast<int>(ball.y)
                   << ") bhumanV=(" << static_cast<int>(theBallModel.estimate.velocity.x()) << ","
                   << static_cast<int>(theBallModel.estimate.velocity.y()) << ") ownV=(" << static_cast<int>(ownVelocity.x())
                   << "," << static_cast<int>(ownVelocity.y()) << ")" << (fitted ? "" : "?")
                   << " cross=" << static_cast<int>(ball.crossY) << "mm in " << ball.crossTime << "s lateral=" << static_cast<int>(lateral));
  }
  const CopaKeeper::Decision decision = copaKeeper.update(static_cast<int>(theFrameInfo.time), ball, lateral, params);

  if(decision.kind != copaKeeperLastAction)
  {
    static const char* names[] = {"watch", "walk", "genuflect left", "genuflect right"};
    OUTPUT_WARNING("[CopaKeeper] " << names[static_cast<int>(decision.kind)]
                   << " ball=(" << ball.x << "," << ball.y << ") vx=" << ball.vx
                   << " cross=" << ball.crossY << "mm in " << ball.crossTime << "s lateral=" << lateral);
    if((decision.kind == CopaKeeper::Action::genuflectLeft || decision.kind == CopaKeeper::Action::genuflectRight) && c.blockMotion != 0)
      SystemCall::say("block");
    copaKeeperLastAction = decision.kind;
  }

  switch(decision.kind)
  {
    case CopaKeeper::Action::genuflectLeft:
    case CopaKeeper::Action::genuflectRight:
      // blockMotion 0: the keeper has already walked into the ball line and blocks with its body.
      if(c.blockMotion == 0)
      {
        theLookAtBallSkill();
        theStandSkill(); // ready stance, knees bent: walking starts at once
        return;
      }
      theLookForwardSkill();
      // Like the RoboCup keeper, go down only from a stand: finish the current step first.
      if(theMotionInfo.executedPhase == MotionPhase::walk)
      {
        theStandSkill();
        return;
      }
      if(c.blockMotion == 1)
      {
        theDiveSkill({.request = MotionRequest::Dive::prepare});
        return;
      }
      theDiveSkill({.request = decision.kind == CopaKeeper::Action::genuflectLeft
                               ? MotionRequest::Dive::squatWideArmsBackLeft
                               : MotionRequest::Dive::squatWideArmsBackRight});
      return;
    case CopaKeeper::Action::walk:
      theLookAtBallSkill();
      theWalkToPointSkill({.target = Pose2f(0_deg, 0.f, decision.walkY),
                           .speed = {c.walkSpeed, c.walkSpeed, c.walkSpeed},
                           .reduceWalkingSpeed = false,
                           .disableAligning = true});
      return;
    case CopaKeeper::Action::watch:
      if(ballFresh)
        theLookAtBallSkill();
      else
        theLookForwardSkill();
      theStandSkill(); // ready stance, knees bent: walking starts at once
      return;
  }
}

void SkillBehaviorControl::executeCopaStriker(bool ballFresh)
{
  const CopaShotConfig& c = copaShotConfig;
  auto enter = [this](CopaStrikerPhase phase)
  {
    copaPhase = phase;
    copaPhaseStart = theFrameInfo.time;
  };

  switch(copaPhase)
  {
    case CopaStrikerPhase::findBall:
      // Striker mode: never walk before the scan. Re-take the mark from the ball in front of the feet.
      theStandSkill({.high = true});
      if(ballFresh)
      {
        theLookAtBallSkill();
        if(!captureCopaMark(false))
          OUTPUT_WARNING("[CopaShot] mark kept from the armed scan, ball not in front of the feet");
      }
      else
      {
        theLookAtAnglesSkill({.pan = 0_deg, .tilt = c.ballScanTilt, .speed = 150_deg});
        if(theFrameInfo.getTimeSince(copaPhaseStart) < c.ballScanMs)
          return;
        OUTPUT_WARNING("[CopaShot] ball not seen at start, mark kept from the armed scan");
      }
      copaSeen = CopaShot::Observations();
      copaLastSnapshot = 0;
      SystemCall::say("looking at goal");
      enter(CopaStrikerPhase::lookGoal);
      return;

    case CopaStrikerPhase::lookGoal:
    {
      theStandSkill({.high = true});
      if(!sweepCopaGoal())
        return;

      CopaShot::Params params;
      params.goalWidth = c.goalWidth;
      params.minGoalSpan = c.minGoalSpan;
      params.maxGoalSpan = c.maxGoalSpan;
      params.postMargin = c.postMargin;
      params.maxAim = c.maxAim;
      params.defaultSide = c.defaultSide;
      // The goal is built from the mark (placement heading); the posts seen correct the heading.
      // Without a matching post, the correction found when armed is kept.
      rebuildCopaGoalFromMark(0.f);
      const Pose2f toRobot = theOdometryData.inverse();
      const Vector2f knownLeft = toRobot * copaLeftPostOdometry, knownRight = toRobot * copaRightPostOdometry;
      CopaShot::PriorGoal prior;
      prior.leftX = knownLeft.x();
      prior.leftY = knownLeft.y();
      prior.rightX = knownRight.x();
      prior.rightY = knownRight.y();
      copaHeadingCorrection = CopaShot::headingCorrection(copaSeen, prior, c.postMatchTolerance, c.maxHeadingCorrection,
                                                          copaPostsMatched);
      if(copaPostsMatched == 0)
        copaHeadingCorrection = copaArmedCorrection;
      const Vector2f mark = toRobot * copaBallOdometry;
      const CopaShot::Shot shot = CopaShot::planShotWithPrior(copaSeen, mark.x(), mark.y(), prior, copaHeadingCorrection, params);
      copaTargetOdometry = theOdometryData * Vector2f(shot.targetX, shot.targetY);
      copaShot = shot;
      copaShotKnown = true;
      copaLeftPostOdometry = theOdometryData * Vector2f(shot.goalLeftX, shot.goalLeftY);
      copaRightPostOdometry = theOdometryData * Vector2f(shot.goalRightX, shot.goalRightY);
      copaKeeperOdometry = theOdometryData * Vector2f(shot.keeperX, shot.keeperY);
      OUTPUT_WARNING("[CopaShot] postSamples=" << static_cast<unsigned>(copaSeen.posts.size())
                  << " postsMatched=" << copaPostsMatched << " headingCorrection=" << Angle(copaHeadingCorrection).toDegrees() << "deg"
                  << " robotSamples=" << static_cast<unsigned>(copaSeen.robots.size())
                  << " keeper=" << (shot.keeperSeen ? "yes" : "no")
                  << " side=" << shot.side << " direction=" << Angle(shot.direction).toDegrees() << "deg"
                  << " target=(" << shot.targetX << "," << shot.targetY << ") reason=" << shot.reason);
      // Like B-Human's penalty taker: aim left with the right foot, right with the left foot
      // (each foot's kick drifts to its own side, away from the post).
      copaKickType = shot.side > 0 ? KickInfo::forwardFastRightLong : KickInfo::forwardFastLeftLong;
      OUTPUT_WARNING("[CopaShot] kick=" << TypeRegistry::getEnumName(copaKickType));
      SystemCall::say(shot.side > 0 ? "aim left" : shot.side < 0 ? "aim right" : "aim center");
      theLookForwardSkill();
      enter(CopaStrikerPhase::align);
      return;
    }

    case CopaStrikerPhase::align:
    {
      // Reach B-Human's kick pose from behind: walk targets are clamped into the box behind the
      // ball (never past the line through the ball, little sideways, little back), the mark is
      // tracked in real time, and only the kick walks into the ball.
      trackCopaMark(ballFresh);
      const KickInfo::Kick& kick = theKickInfo[copaKickType];
      const Vector2f mark = theOdometryData.inverse() * copaBallOdometry;
      const Vector2f target = theOdometryData.inverse() * copaTargetOdometry;
      const Angle direction = std::clamp(Angle((target - mark).angle()), Angle(-c.maxAim), Angle(c.maxAim));
      const CopaShot::Pose pre = CopaShot::preKickPose(mark.x(), mark.y(), direction, kick.ballOffset.x(), kick.ballOffset.y(),
                                                       kick.rotationOffset, c.preKickClearance);
      copaFootClearance = CopaShot::footClearance(mark.x(), mark.y(), theBallSpecification.radius, copaFeet());
      copaMinFootClearance = std::min(copaMinFootClearance, copaFootClearance);
      copaPreKickRelative = clampCopaTarget(Pose2f(pre.rotation, pre.x, pre.y));
      copaPreKickValid = true;
      theLookAtBallSkill();
      const Pose2f& goal = copaPreKickRelative;
      const bool aligned = std::abs(goal.translation.x()) < c.alignTolerance && std::abs(goal.translation.y()) < c.alignTolerance
                           && std::abs(goal.rotation) < c.alignAngleTolerance;
      if(aligned || theFrameInfo.getTimeSince(copaPhaseStart) > c.alignTimeoutMs)
      {
        OUTPUT_WARNING("[CopaShot] aligned=" << (aligned ? "yes" : "timeout") << " remaining=(" << static_cast<int>(goal.translation.x())
                       << "," << static_cast<int>(goal.translation.y()) << "," << goal.rotation.toDegrees() << "deg) mark=("
                       << static_cast<int>(mark.x()) << "," << static_cast<int>(mark.y()) << ") trackShift="
                       << static_cast<int>(copaTrackShift.norm()) << "mm footClearance=" << static_cast<int>(copaFootClearance)
                       << "mm minFootClearance=" << static_cast<int>(copaMinFootClearance) << "mm");
        theStandSkill();
        enter(CopaStrikerPhase::kick);
        return;
      }
      // Too close to a sole: only step straight back (still clamped into the box).
      copaWalkTargetRelative = copaFootClearance < c.minFootClearance ? clampCopaTarget(Pose2f(0_deg, -60.f, 0.f)) : goal;
      copaWalkValid = true;
      theWalkToPointSkill({.target = copaWalkTargetRelative, .speed = {0.4f, 0.4f, 0.4f},
                           .disableObstacleAvoidance = true});
      return;
    }

    case CopaStrikerPhase::kick:
    {
      trackCopaMark(ballFresh);
      if(!ballFresh && theFrameInfo.getTimeSince(copaPhaseStart) > 3000)
      {
        enter(CopaStrikerPhase::align); // re-align from the tracked mark; never walk to "find" the ball
        return;
      }
      const Vector2f mark = theOdometryData.inverse() * copaBallOdometry;
      const Vector2f target = theOdometryData.inverse() * copaTargetOdometry;
      const Angle direction = std::clamp(Angle((target - mark).angle()), Angle(-c.maxAim), Angle(c.maxAim));
      theLookAtBallSkill();
      theWalkToBallAndKickSkill({.targetDirection = direction,
                                 .kickType = copaKickType,
                                 .alignPrecisely = KickPrecision::precise,
                                 .speed = {0.4f, 0.4f, 0.4f}});
      return;
    }

    default: // armed-only phases
      enter(CopaStrikerPhase::findBall);
      return;
  }
}

void SkillBehaviorControl::executeRequest()
{
  if(theSkillRequest.skill == SkillRequest::none)
  {
    if(theFrameInfo.getTimeSince(timeWhenAnnouncedEmptySkillRequest) > 5000)
    {
      ANNOTATION("Behavior", "Skill request is empty");
      SystemCall::say("Skill request is empty");
      timeWhenAnnouncedEmptySkillRequest = theFrameInfo.time;
    }
  }
  else if(theSkillRequest.skill == SkillRequest::shoot ||
          theSkillRequest.skill == SkillRequest::pass ||
          theSkillRequest.skill == SkillRequest::dribble)
  {
    thePlayBallSkill();
  }
  else
  {
    for(const Teammate& teammate : theTeamData.teammates)
    {
      if(teammate.theBehaviorStatus.passTarget == theGameState.playerNumber)
      {
        theBehaviorStatus.passOrigin = teammate.number;
        theReceivePassSkill({.playerNumber = teammate.number});
        return;
      }
    }

    switch(theSkillRequest.skill)
    {
      case SkillRequest::stand:
        theStandSkill();
        theLookActiveSkill({.withBall = true});
        break;
      case SkillRequest::walk:
      {
        const Pose2f targetPose = theRobotPose.inverse() * theSkillRequest.target;
        const bool rlTeamUsesObstacleAvoidance = RLSharedStateBridge::isEnabledForTeam(theGameState.ownTeam.number);
        const bool goalkeeperMode = theGameState.isGoalkeeper() && !rlTeamUsesObstacleAvoidance;
        if((theFieldBall.ballWasSeen(7000) || theTeammatesBallModel.isValid) && theFieldBall.isBallPositionConsistentWithGameState())
        {
          theWalkToPointObstacleSkill({.target = theRobotPose.inverse() * theSkillRequest.target,
                                       .rough = goalkeeperMode,
                                       .disableObstacleAvoidance = goalkeeperMode,
                                       .targetOfInterest = theFieldBall.recentBallPositionRelative()}); // TODO: set the right parameters and occasionally use WalkPotentialField
          if(theMotionInfo.isMotion(MotionPhase::stand))
            theLookActiveSkill({.withBall = true});
          else
            theLookAtBallAndTargetSkill({.startTarget = false,
                                         .walkingDirection = targetPose.translation,
                                         .ballPositionAngle = theFieldBall.recentBallPositionRelative().angle()});
        }
        else
        {
          theWalkToPointObstacleSkill({.target = theRobotPose.inverse() * theSkillRequest.target,
                                       .rough = goalkeeperMode,
                                       .disableObstacleAvoidance = goalkeeperMode}); // TODO: set the right parameters and occasionally use WalkPotentialField
          theLookActiveSkill({.withBall = true});
        }
        break;
      }
      case SkillRequest::block:
        theBlockSkill({.target = theRobotPose.inverse() * theSkillRequest.target.translation});
        break;
      case SkillRequest::mark:
        theMarkSkill({.target = theRobotPose.inverse() * theSkillRequest.target.translation});
        break;
      case SkillRequest::observe:
        theObservePointSkill({.target = theRobotPose.inverse() * theSkillRequest.target.translation});
        break;
      case SkillRequest::interceptBall:
        theInterceptBallSkill({.interceptionMethods = theSkillRequest.interceptionMethods,
                               .allowDive = theSkillRequest.allowDive});
        break;
      case SkillRequest::keeperDive:
        theDiveSkill({.request = theSkillRequest.diveRequest});
        break;
      default:
        FAIL("Unknown skill request.");
    }
  }
}
