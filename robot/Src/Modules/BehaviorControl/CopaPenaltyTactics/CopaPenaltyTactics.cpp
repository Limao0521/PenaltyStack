/**
 * @file CopaPenaltyTactics.cpp
 *
 * @author PenaltyStack
 */

#include "CopaPenaltyTactics.h"
#include "Debugging/Annotation.h"
#include "Platform/SystemCall.h"
#include "Streaming/InStreams.h"
#include "Streaming/Output.h"
#include <algorithm>
#include <cmath>
#include <iostream>

MAKE_MODULE(CopaPenaltyTactics);

using CopaPenalty::BallEstimate;
using CopaPenalty::BallMeasurement;
using CopaPenalty::BallTracker;
using CopaPenalty::Defense;
using CopaPenalty::DenseLayer;
using CopaPenalty::Field;
using CopaPenalty::LibraryMode;
using CopaPenalty::LibraryOutput;
using CopaPenalty::MlpPolicyParameters;
using CopaPenalty::Observation;
using CopaPenalty::Pose;
using CopaPenalty::StrikerAction;
using CopaPenalty::Vec2;
using CopaPenalty::VisionKeeperSettings;
using CopaPenalty::WorldState;

namespace
{
  DenseLayer toLayer(const CopaDenseLayerConfig& config)
  {
    DenseLayer layer;
    layer.rows = config.rows;
    layer.cols = config.cols;
    layer.weights.assign(config.weights.begin(), config.weights.end());
    layer.bias.assign(config.bias.begin(), config.bias.end());
    return layer;
  }

  MotionRequest::Dive::Request toDive(Defense defense, bool allowJumps)
  {
    switch(defense)
    {
      case Defense::center: return MotionRequest::Dive::squatArmsBackLeft;
      case Defense::centerRight: return MotionRequest::Dive::squatArmsBackRight;
      case Defense::wideLeft: return MotionRequest::Dive::squatWideArmsBackLeft;
      case Defense::wideRight: return MotionRequest::Dive::squatWideArmsBackRight;
      case Defense::left: return allowJumps ? MotionRequest::Dive::jumpLeft : MotionRequest::Dive::squatWideArmsBackLeft;
      case Defense::right: return allowJumps ? MotionRequest::Dive::jumpRight : MotionRequest::Dive::squatWideArmsBackRight;
    }
    return MotionRequest::Dive::squatArmsBackLeft;
  }
}

CopaPenaltyTactics::CopaPenaltyTactics()
{
  loadPolicies();
}

void CopaPenaltyTactics::loadPolicies()
{
  loadedStrikerPolicy = strikerPolicy;
  loadedKeeperPolicy = keeperPolicy;
  mlp.reset();
  if(strikerPolicy != "scripted")
  {
    InMapFile stream("NeuralNets/CopaPenalty/" + strikerPolicy + ".cfg");
    if(stream.exists())
    {
      CopaMlpPolicyConfig config;
      stream >> config;
      MlpPolicyParameters parameters;
      parameters.name = config.name;
      parameters.history = config.history;
      parameters.obsMean = config.obsMean;
      parameters.obsVar = config.obsVar;
      parameters.clipObs = config.clipObs;
      parameters.epsilon = config.epsilon;
      for(const CopaDenseLayerConfig& layer : config.hidden)
        parameters.hidden.push_back(toLayer(layer));
      parameters.action = toLayer(config.action);
      if(parameters.valid())
        mlp.emplace(std::move(parameters));
    }
    if(!mlp)
    {
      OUTPUT_ERROR("CopaPenaltyTactics: striker policy " << strikerPolicy << " missing or invalid; using scripted");
      SystemCall::say("striker policy missing");
    }
  }

  VisionKeeperSettings settings;
  settings.cornerY = keeperCornerY;
  settings.diveLeadS = keeperDiveLead;
  settings.minIncomingSpeed = keeperMinIncomingSpeed;
  settings.confirmFrames = keeperConfirmFrames;
  settings.history = keeperHistory;
  settings.maxBallAge = maxBallAge;
  settings.sigmaMargin = keeperSigmaMargin;
  settings.goalLineX = keeperGoalLineX;
  if(!settings.valid())
  {
    OUTPUT_ERROR("CopaPenaltyTactics: invalid keeper settings; using vision_keeper_v2 defaults");
    settings = VisionKeeperSettings();
  }
  centerGuard = keeperPolicy == "center_guard";
  if(keeperPolicy != "vision_v2_balanced" && keeperPolicy != "vision_v2_original" && !centerGuard)
  {
    OUTPUT_ERROR("CopaPenaltyTactics: unknown keeper policy " << keeperPolicy << "; using vision_v2_balanced");
    SystemCall::say("keeper policy missing");
  }
  library.emplace(keeperPolicy == "vision_v2_original" || centerGuard ? LibraryMode::original : LibraryMode::balanced, settings);
  tracker = BallTracker(maxBallAge);
}

Field CopaPenaltyTactics::geometry() const
{
  Field field;
  field.goalLineX = field.penaltyMarkX - penaltyMarkToGoalLine / 1000.;
  field.postY = goalWidth / 2000.;
  field.strikerStart = {field.penaltyMarkX + strikerBehindMark / 1000., 0., CopaPenalty::pi};
  field.keeperStart = {field.goalLineX + keeperOffsetFromGoalLine / 1000., 0., 0.};
  return field;
}

Pose CopaPenaltyTactics::odometryPose() const
{
  return {theOdometryData.translation.x() / 1000., theOdometryData.translation.y() / 1000.,
          static_cast<double>(theOdometryData.rotation)};
}

Pose CopaPenaltyTactics::fieldPose() const
{
  const Field field = geometry();
  const Pose& start = theCopaPenaltyState.role == CopaPenaltyState::striker ? field.strikerStart : field.keeperStart;
  return CopaPenalty::compose(start, CopaPenalty::between(odometryAtStart, odometryPose()));
}

void CopaPenaltyTactics::update(CopaPenaltyDecision& decision)
{
  if(strikerPolicy != loadedStrikerPolicy || keeperPolicy != loadedKeeperPolicy)
    loadPolicies();

  decision = CopaPenaltyDecision();
  const bool isStriker = theCopaPenaltyState.role == CopaPenaltyState::striker;
  decision.policy = isStriker ? (mlp ? strikerPolicy : std::string("scripted")) : keeperPolicy;
  if(!theCopaPenaltyState.isPlaying())
  {
    decision.command = CopaPenaltyDecision::idle;
    decision.reason = TypeRegistry::getEnumName(theCopaPenaltyState.phase);
    return;
  }
  if(theCopaPenaltyState.attempt != routineAttempt)
    startRoutine();
  const bool newFrame = trackBall(decision);
  if(isStriker)
    striker(decision);
  else
    keeper(decision, newFrame);
}

void CopaPenaltyTactics::startRoutine()
{
  routineAttempt = theCopaPenaltyState.attempt;
  odometryAtStart = odometryPose();
  tracker.reset();
  observations.reset();
  if(mlp)
    mlp->reset();
  if(library)
    library->reset();
  ball = BallEstimate();
  lastBallField.reset();
  lastSeenCamera.reset();
  lastSeenTime = 0;
  lastDecision = 0;
  shot = StrikerAction::scripted();
  kickReleased = false;
  timeWhenKickReleased = timeWhenKicked = 0;
  lastKickTimestampAtRelease = theMotionInfo.lastKickTimestamp;
  travelExceeded = false;
  dove = false;
  timeWhenDove = 0;
}

bool CopaPenaltyTactics::trackBall(CopaPenaltyDecision& decision)
{
  const double now = theFrameInfo.time / 1000.;
  const Pose odometry = odometryPose();
  std::optional<BallMeasurement> measurement;
  // Cognition alternates upper and lower camera frames. When only one camera sees the ball, the
  // other camera's empty frames must not interrupt the track (vision_keeper_v2 needs consecutive
  // incoming measurements to confirm a shot), so they are skipped while the sighting is recent.
  const bool seen = theBallPercept.status == BallPercept::seen;
  const bool otherCamera = lastSeenCamera && *lastSeenCamera != theCameraInfo.camera;
  bool newFrame = true;
  if(!seen && otherCamera && theFrameInfo.getTimeSince(lastSeenTime) < otherCameraGraceMs)
    newFrame = false;
  if(seen)
  {
    lastSeenCamera = theCameraInfo.camera;
    lastSeenTime = theFrameInfo.time;
    BallMeasurement value;
    value.position = {theBallPercept.positionOnField.x() / 1000., theBallPercept.positionOnField.y() / 1000.};
    for(int i = 0; i < 2; ++i)
      for(int j = 0; j < 2; ++j)
        value.covariance[i][j] = theBallPercept.covarianceOnField(i, j) / 1e6;
    // Symmetrize: float covariance from the percept may differ in the last bit.
    value.covariance[0][1] = value.covariance[1][0] = (value.covariance[0][1] + value.covariance[1][0]) / 2.;
    measurement = value;
  }
  if(newFrame)
    tracker.update(now, now, measurement, odometry);
  ball = tracker.estimate(now, odometry);
  decision.ballValid = ball.positionValid;
  decision.ballPosition = Vector2f(static_cast<float>(ball.position.x * 1000.), static_cast<float>(ball.position.y * 1000.));
  decision.ballVelocity = Vector2f(static_cast<float>(ball.velocity.x * 1000.), static_cast<float>(ball.velocity.y * 1000.));
  const unsigned ageMs = static_cast<unsigned>(std::min(ball.ageS, 40.) * 1000.);
  decision.ballTimeWhenLastSeen = theFrameInfo.time > ageMs ? theFrameInfo.time - ageMs : 0;
  return newFrame;
}

StrikerAction CopaPenaltyTactics::decideShot(const Field& field, const Pose& pose)
{
  if(!mlp)
    return StrikerAction::scripted();
  const double t = theFrameInfo.getTimeSince(theCopaPenaltyState.timeWhenStarted) / 1000.;
  if(ball.positionValid)
    lastBallField = CopaPenalty::toField(ball.position, pose);
  WorldState state;
  state.time = t;
  state.remainingTime = field.penaltySeconds - t;
  state.ball = lastBallField ? *lastBallField : Vec2{field.penaltyMarkX, 0.};
  state.ballVisible = ball.visible;
  state.robot = pose;
  state.opponent = field.keeperStart; // the keeper stands on its goal line, facing the striker (rule 3.3.5)
  state.torsoRoll = theInertialData.angle.x();
  state.torsoPitch = theInertialData.angle.y();
  state.gyro = {theInertialData.gyro.x(), theInertialData.gyro.y(), theInertialData.gyro.z()};
  const Observation observation = observations.build(state, field);
  return StrikerAction::fromNormalized((*mlp)(CopaPenalty::clipObservation(observation)));
}

void CopaPenaltyTactics::striker(CopaPenaltyDecision& decision)
{
  const Field field = geometry();
  const Pose pose = fieldPose();

  if(!kickReleased && std::hypot(pose.x - field.strikerStart.x, pose.y - field.strikerStart.y) > maxStrikerTravel / 1000.)
    travelExceeded = true;
  if(!kickReleased && (!lastDecision || theFrameInfo.getTimeSince(lastDecision) >= decisionPeriodMs))
  {
    lastDecision = theFrameInfo.time;
    shot = decideShot(field, pose);
  }
  decision.targetY = static_cast<float>(shot.targetY);
  decision.trigger = static_cast<float>(shot.trigger);
  decision.kickType = shot.rightFoot ? KickInfo::forwardFastRight : KickInfo::forwardFastLeft;
  decision.kickLength = static_cast<float>(shot.kickLengthMm());
  decision.walkSpeed = walkSpeedRatio;

  // Aim point on the goal line; the striker's left is field -y (it looks towards -x).
  const Vec2 aim = CopaPenalty::toRobot({field.goalLineX, -shot.targetY}, pose.translation(), pose.yaw);
  const double direction = std::atan2(aim.y - ball.position.y, aim.x - ball.position.x);
  decision.kickDirection = static_cast<float>(direction);
  const double distance = std::hypot(ball.position.x, ball.position.y);

  if(timeWhenKicked)
  {
    decision.command = theFrameInfo.getTimeSince(timeWhenKicked) >= recoverMs ? CopaPenaltyDecision::done
                                                                            : CopaPenaltyDecision::recover;
    decision.reason = "kicked";
    return;
  }
  if(kickReleased)
  {
    if(theMotionInfo.lastKickTimestamp > lastKickTimestampAtRelease)
    {
      timeWhenKicked = theFrameInfo.time;
      decision.command = CopaPenaltyDecision::recover;
      decision.reason = "kicked";
      ANNOTATION("CopaPenaltyTactics", "kick done");
      std::cout << "[CopaPenalty] t=" << theFrameInfo.time << " kick done" << std::endl;
    }
    else if(theFrameInfo.getTimeSince(timeWhenKickReleased) > kickTimeoutMs)
    {
      timeWhenKicked = theFrameInfo.time;
      decision.command = CopaPenaltyDecision::recover;
      decision.reason = "kick_timeout";
      ANNOTATION("CopaPenaltyTactics", "kick timeout");
      std::cout << "[CopaPenalty] t=" << theFrameInfo.time << " kick timeout" << std::endl;
    }
    else
    {
      decision.command = CopaPenaltyDecision::kick;
      decision.reason = "kicking";
    }
    return;
  }
  if(travelExceeded)
  {
    decision.command = CopaPenaltyDecision::done;
    decision.reason = "travel_limit";
    return;
  }
  if(!ball.positionValid)
  {
    decision.command = CopaPenaltyDecision::search;
    decision.reason = ball.reason;
    return;
  }
  if(shot.fires() && distance <= kickReleaseRange / 1000.)
  {
    kickReleased = true;
    timeWhenKickReleased = theFrameInfo.time;
    lastKickTimestampAtRelease = theMotionInfo.lastKickTimestamp;
    decision.command = CopaPenaltyDecision::kick;
    decision.reason = "released";
    ANNOTATION("CopaPenaltyTactics", "kick released target_y=" << shot.targetY << " right=" << shot.rightFoot
               << " length=" << shot.kickLengthMm());
    std::cout << "[CopaPenalty] t=" << theFrameInfo.time << " kick released policy=" << decision.policy << " target_y="
              << shot.targetY << " foot=" << (shot.rightFoot ? "right" : "left") << " length=" << shot.kickLengthMm()
              << " ball=(" << ball.position.x << "," << ball.position.y << ")" << std::endl;
    return;
  }
  // Walk to the pose behind the ball facing the aim; never turn in place on top of the ball.
  const double d = kickPoseDistance / 1000.;
  decision.approachPose = Pose2f(static_cast<float>(direction),
                                 static_cast<float>((ball.position.x - d * std::cos(direction)) * 1000.),
                                 static_cast<float>((ball.position.y - d * std::sin(direction)) * 1000.));
  decision.command = CopaPenaltyDecision::approach;
  decision.reason = shot.fires() ? "approaching" : "waiting_trigger";
}

void CopaPenaltyTactics::keeper(CopaPenaltyDecision& decision, bool newFrame)
{
  const Field field = geometry();
  const double t = theFrameInfo.getTimeSince(theCopaPenaltyState.timeWhenStarted) / 1000.;
  if(dove)
  {
    const bool holding = theFrameInfo.getTimeSince(timeWhenDove) < diveHoldMs;
    decision.command = holding ? CopaPenaltyDecision::dive : CopaPenaltyDecision::guard;
    decision.diveRequest = diveRequest;
    decision.trigger = 1.f;
    decision.reason = holding ? "committed" : "after_defense";
    return;
  }
  if(!newFrame)
  {
    decision.command = CopaPenaltyDecision::guard;
    decision.reason = library->getDecision().reason;
    return;
  }
  const LibraryOutput output = (*library)(t, t, std::max(0., field.penaltySeconds - t), ball);
  decision.trigger = static_cast<float>(output.trigger);
  decision.reason = library->getDecision().reason;
  if(library->getDecision().crossingY)
    decision.targetY = static_cast<float>(*library->getDecision().crossingY);
  if(output.fires())
  {
    const Defense defense = centerGuard ? Defense::center : output.defense;
    dove = true;
    timeWhenDove = theFrameInfo.time;
    diveRequest = toDive(defense, allowJumpDives);
    decision.command = CopaPenaltyDecision::dive;
    decision.diveRequest = diveRequest;
    ANNOTATION("CopaPenaltyTactics", "defense " << CopaPenalty::defenseName(defense));
    std::cout << "[CopaPenalty] t=" << theFrameInfo.time << " defense " << CopaPenalty::defenseName(defense)
              << " policy=" << keeperPolicy << " crossing_y="
              << library->getDecision().crossingY.value_or(0.) << " time_to_line="
              << library->getDecision().timeToLine.value_or(0.) << std::endl;
    return;
  }
  decision.command = CopaPenaltyDecision::guard;
}
