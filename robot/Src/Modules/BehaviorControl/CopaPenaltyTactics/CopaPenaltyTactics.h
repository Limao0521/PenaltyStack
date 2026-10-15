/**
 * @file CopaPenaltyTactics.h
 *
 * Tactical layer of the Copa NAO CCM MX 2026 penalty routine. It runs the policies
 * validated in PenaltyStack's MuJoCo + B-Human simulation, ported to C++ in
 * Tools/CopaPenalty/CopaPolicies.h and checked step by step against Python:
 *
 *  - striker: rl_obs_v1 observation every 100 ms (like the simulator's FSM period),
 *    deterministic PPO actor (striker_v2, striker_preserving_v4, ...) or the scripted
 *    shot; the kick is released once and B-Human walks to the ball and kicks.
 *  - keeper: ball tracker + vision_keeper_v2 with the six original defenses
 *    (balanced/original) or a fixed central guard.
 *
 * Poses come from odometry since the start: the striker is placed 20 cm behind the
 * penalty mark facing the goal and the keeper on its goal line (rule 3.3.5). The ball
 * is tracked from this module's own BallPercept stream, independent of field-line
 * localization (the competition floor is wood with black lines).
 *
 * @author PenaltyStack
 */

#pragma once

#include "Framework/Module.h"
#include "Representations/BehaviorControl/CopaPenaltyDecision.h"
#include "Representations/BehaviorControl/CopaPenaltyState.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Infrastructure/FrameInfo.h"
#include "Representations/MotionControl/MotionInfo.h"
#include "Representations/MotionControl/OdometryData.h"
#include "Representations/Perception/BallPercepts/BallPercept.h"
#include "Representations/Sensing/InertialData.h"
#include "Streaming/AutoStreamable.h"
#include "Tools/CopaPenalty/CopaPolicies.h"
#include <optional>
#include <string>
#include <vector>

/** One dense layer of an exported actor (Config/NeuralNets/CopaPenalty/<name>.cfg). */
STREAMABLE(CopaDenseLayerConfig,
{,
  (unsigned)(0) rows,
  (unsigned)(0) cols,
  (std::vector<float>) weights,
  (std::vector<float>) bias,
});

/** A deterministic SB3 actor exported by tools/export_robot_policies.py. */
STREAMABLE(CopaMlpPolicyConfig,
{,
  (std::string) name,
  (unsigned)(1) history,
  (std::vector<double>) obsMean,
  (std::vector<double>) obsVar,
  (double)(10.) clipObs,
  (double)(1e-8) epsilon,
  (std::vector<CopaDenseLayerConfig>) hidden,
  (CopaDenseLayerConfig) action,
});

MODULE(CopaPenaltyTactics,
{,
  REQUIRES(BallPercept),
  REQUIRES(CameraInfo),
  REQUIRES(CopaPenaltyState),
  REQUIRES(FrameInfo),
  REQUIRES(InertialData),
  REQUIRES(MotionInfo),
  REQUIRES(OdometryData),
  PROVIDES(CopaPenaltyDecision),
  LOADS_PARAMETERS(
  {,
    (std::string) strikerPolicy,        /**< scripted | striker_v2 | striker_preserving_v4 | striker_visual_v3 | striker_v1 */
    (std::string) keeperPolicy,         /**< vision_v2_balanced | vision_v2_original | center_guard */
    (float) penaltyMarkToGoalLine,      /**< mm (rules 3.3.4: 1300). */
    (float) strikerBehindMark,          /**< mm (rules 3.3.5: 200). */
    (float) goalWidth,                  /**< mm (rules 3.3.4: 1500). */
    (float) keeperOffsetFromGoalLine,   /**< mm from the goal line to the keeper's feet centre. */
    (int) decisionPeriodMs,             /**< Striker policy period (simulator FSM: 100 ms). */
    (float) kickReleaseRange,           /**< mm: the kick is released only with the ball this close. */
    (float) kickPoseDistance,           /**< mm behind the ball for the approach pose. */
    (float) maxStrikerTravel,           /**< mm from the start; beyond it the striker stops (never approach the keeper). */
    (int) kickTimeoutMs,                /**< Walk-to-ball-and-kick must finish within this time. */
    (int) recoverMs,                    /**< Standing time after the kick before reporting done. */
    (float) maxBallAge,                 /**< s, tracker freshness (vision_state_v1: 0.25). */
    (int) otherCameraGraceMs,           /**< A frame of the other camera without ball is skipped this soon after a sighting. */
    (float) keeperCornerY,              /**< m, vision_keeper_v2 corner threshold. */
    (float) keeperDiveLead,             /**< s, vision_keeper_v2 dive lead time. */
    (float) keeperMinIncomingSpeed,     /**< m/s. */
    (int) keeperConfirmFrames,
    (int) keeperHistory,
    (float) keeperSigmaMargin,
    (float) keeperGoalLineX,            /**< m, robot-relative line to defend (behind the feet). */
    (int) diveHoldMs,                   /**< How long the dive request is held before standing up again. */
    (bool) allowJumpDives,              /**< false replaces the lateral jumps by the wide squat guards (no fall). */
    (float) walkSpeedRatio,             /**< Walking speed ratio of the maximum for approach and kick. */
  }),
});

class CopaPenaltyTactics : public CopaPenaltyTacticsBase
{
public:
  CopaPenaltyTactics();

private:
  void update(CopaPenaltyDecision& decision) override;

  void loadPolicies();
  void startRoutine();
  CopaPenalty::Field geometry() const;
  CopaPenalty::Pose odometryPose() const;
  CopaPenalty::Pose fieldPose() const;
  /** @return Whether this frame was fed to the tracker (false: other camera, ball seen by the active one). */
  bool trackBall(CopaPenaltyDecision& decision);
  void striker(CopaPenaltyDecision& decision);
  void keeper(CopaPenaltyDecision& decision, bool newFrame);
  CopaPenalty::StrikerAction decideShot(const CopaPenalty::Field& field, const CopaPenalty::Pose& pose);

  std::string loadedStrikerPolicy;
  std::string loadedKeeperPolicy;
  std::optional<CopaPenalty::MlpPolicy> mlp;
  std::optional<CopaPenalty::LibraryKeeper> library;
  bool centerGuard = false;

  CopaPenalty::BallTracker tracker;
  CopaPenalty::ObservationBuilder observations;
  CopaPenalty::BallEstimate ball;
  std::optional<CopaPenalty::Vec2> lastBallField;
  std::optional<CameraInfo::Camera> lastSeenCamera;
  unsigned lastSeenTime = 0;

  unsigned routineAttempt = 0;
  CopaPenalty::Pose odometryAtStart;
  unsigned lastDecision = 0;
  CopaPenalty::StrikerAction shot;
  bool kickReleased = false;
  unsigned timeWhenKickReleased = 0;
  unsigned lastKickTimestampAtRelease = 0;
  unsigned timeWhenKicked = 0;
  bool travelExceeded = false;
  bool dove = false;
  unsigned timeWhenDove = 0;
  MotionRequest::Dive::Request diveRequest = MotionRequest::Dive::prepare;
};
