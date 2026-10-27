/**
 * @file SkillBehaviorControl.cpp
 *
 * This file implements the behavior control for the skill layer.
 *
 * @author Arne Hasselbring
 */

#include "SkillBehaviorControl.h"
#include "Python/Controller/RLSharedState.h"
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
  if(state != previousCopaState)
  {
    OUTPUT_TEXT("CopaPenalty state=" << static_cast<int>(state) << " role=" << (keeper ? "keeper" : "striker"));
    const char* role = keeper ? "goalkeeper" : "striker";
    const char* name = state == CopaPenaltyState::waiting ? "armed" : state == CopaPenaltyState::running ? "start" : "stop";
    SystemCall::say((std::string(role) + " mode " + name).c_str());
    if(state == CopaPenaltyState::running)
    {
      copaLastKick = theMotionInfo.lastKickTimestamp;
      copaKeeper.reset();
      copaKeeperStart = theOdometryData;
      copaKeeperLastAction = CopaKeeper::Action::watch;
      copaBallTrack.reset();
      copaPhase = CopaStrikerPhase::findBall;
      copaPhaseStart = theFrameInfo.time;
    }
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

  if(!theCameraStatus.ok || state != CopaPenaltyState::running)
  {
    if(!theCameraStatus.ok)
      copaState.finish();
    theLookForwardSkill();
    theStandSkill({.high = true});
    return;
  }

  const bool ballFresh = theBallModel.timeWhenLastSeen != 0 && theFrameInfo.getTimeSince(theBallModel.timeWhenLastSeen) < 500;
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
  const Vector2f ball = theBallModel.estimate.position;
  auto enter = [this](CopaStrikerPhase phase)
  {
    copaPhase = phase;
    copaPhaseStart = theFrameInfo.time;
  };

  switch(copaPhase)
  {
    case CopaStrikerPhase::findBall:
      if(!ballFresh)
      {
        theLookLeftAndRightSkill({.maxPan = 30_deg, .tilt = 5.7_deg, .speed = 30_deg});
        theStandSkill({.high = true});
        return;
      }
      theLookAtBallSkill();
      if(ball.norm() > c.approachDistance)
      {
        // Keep facing the goal: walk straight up behind the ball, no turning.
        theWalkToPointSkill({.target = Pose2f(0_deg, ball.x() - c.standOffDistance, ball.y()),
                             .speed = {0.5f, 0.5f, 0.5f}});
        return;
      }
      // One head and one motion skill per frame: the sweep starts next frame.
      theStandSkill({.high = true});
      copaSeen = CopaShot::Observations();
      copaLastSnapshot = 0;
      SystemCall::say("looking at goal");
      enter(CopaStrikerPhase::lookGoal);
      return;

    case CopaStrikerPhase::lookGoal:
    {
      theStandSkill({.high = true});
      const Angle pans[3] = {c.sweepPan, 0_deg, -c.sweepPan};
      const int elapsed = static_cast<int>(theFrameInfo.getTimeSince(copaPhaseStart));
      const int step = elapsed / std::max(1, c.sweepStepMs);
      if(step < 3)
      {
        theLookAtAnglesSkill({.pan = pans[step], .tilt = c.headTilt, .speed = 150_deg});
        const CopaVisionSnapshot snapshot = CopaVisionShared::instance().read();
        const unsigned settled = copaPhaseStart + static_cast<unsigned>(step * c.sweepStepMs + c.settleMs);
        if(snapshot.time != copaLastSnapshot && snapshot.time >= settled)
        {
          copaLastSnapshot = snapshot.time;
          copaSeen.posts.insert(copaSeen.posts.end(), snapshot.postBearings.begin(), snapshot.postBearings.end());
          copaSeen.robots.insert(copaSeen.robots.end(), snapshot.robotBearings.begin(), snapshot.robotBearings.end());
        }
        return;
      }

      CopaShot::Params params;
      params.goalWidth = c.goalWidth;
      params.minGoalSpan = c.minGoalSpan;
      params.maxGoalSpan = c.maxGoalSpan;
      params.postMargin = c.postMargin;
      params.maxAim = c.maxAim;
      params.defaultSide = c.defaultSide;
      const CopaShot::Shot shot = CopaShot::planShot(copaSeen, ball.x(), ball.y(), params, c.expectedGoalDistance);
      copaTargetOdometry = theOdometryData * Vector2f(shot.targetX, shot.targetY);
      OUTPUT_WARNING("[CopaShot] postSamples=" << static_cast<unsigned>(copaSeen.posts.size())
                  << " robotSamples=" << static_cast<unsigned>(copaSeen.robots.size())
                  << " goal=" << (shot.goalSeen ? "yes" : "no") << " keeper=" << (shot.keeperSeen ? "yes" : "no")
                  << " side=" << shot.side << " direction=" << Angle(shot.direction).toDegrees() << "deg"
                  << " target=(" << shot.targetX << "," << shot.targetY << ") reason=" << shot.reason);
      SystemCall::say(!shot.goalSeen ? "no goal, straight" : shot.side > 0 ? "aim left" : shot.side < 0 ? "aim right" : "aim center");
      theLookForwardSkill();
      enter(CopaStrikerPhase::kick);
      return;
    }

    case CopaStrikerPhase::kick:
    {
      if(!ballFresh && theFrameInfo.getTimeSince(copaPhaseStart) > 3000)
      {
        enter(CopaStrikerPhase::findBall);
        return;
      }
      const Vector2f target = theOdometryData.inverse() * copaTargetOdometry;
      const Angle direction = std::clamp(Angle((target - ball).angle()), Angle(-c.maxAim), Angle(c.maxAim));
      theLookAtBallSkill();
      theWalkToBallAndKickSkill({.targetDirection = direction,
                                 .kickType = KickInfo::forwardFastLeftLong,
                                 .alignPrecisely = KickPrecision::precise,
                                 .speed = {0.5f, 0.5f, 0.5f}});
      return;
    }
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
