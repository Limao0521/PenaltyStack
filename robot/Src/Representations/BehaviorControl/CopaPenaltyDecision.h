/**
 * @file CopaPenaltyDecision.h
 *
 * Tactical decision of the penalty routine, provided by CopaPenaltyTactics and executed
 * by the option HandleCopaNAOPenalty with B-Human's own skills (walk, kick, dive, head).
 * The tactical layer chooses what to do; B-Human executes every motion.
 *
 * @author PenaltyStack
 */

#pragma once

#include "Representations/Configuration/KickInfo.h"
#include "Representations/MotionControl/MotionRequest.h"
#include "Math/Angle.h"
#include "Math/Eigen.h"
#include "Math/Pose2f.h"
#include "Streaming/AutoStreamable.h"
#include "Streaming/Enum.h"
#include <string>

STREAMABLE(CopaPenaltyDecision,
{
  ENUM(Command,
  {,
    idle,     /**< Not playing: stand still, head forward. */
    search,   /**< Striker without a fresh ball: stand and scan with the head. */
    approach, /**< Striker: walk to the pose behind the ball, facing the aim point. */
    kick,     /**< Striker: B-Human walk-to-ball-and-kick with the selected kick, exactly once. */
    recover,  /**< Striker after the kick: stand and never follow the ball. */
    done,     /**< Striker: routine complete. */
    guard,    /**< Keeper: stand and track the ball. */
    dive,     /**< Keeper: the committed defense (original B-Human dive/squat motion). */
  });

  bool isStriker() const { return command == search || command == approach || command == kick || command == recover || command == done; },

  (Command)(idle) command,
  (bool)(false) ballValid,                                   /**< The tracked ball is fresh (vision_state rules). */
  (Vector2f)(Vector2f::Zero()) ballPosition,                 /**< Robot-relative tracked ball (mm). */
  (Vector2f)(Vector2f::Zero()) ballVelocity,                 /**< Robot-relative ball velocity (mm/s). */
  (unsigned)(0) ballTimeWhenLastSeen,                        /**< Timestamp of the last accepted ball measurement. */
  (Pose2f) approachPose,                                     /**< Robot-relative target pose for `approach`. */
  (Angle)(0_deg) kickDirection,                              /**< Robot-relative direction from the ball to the aim point. */
  (KickInfo::KickType)(KickInfo::forwardFastLeft) kickType,  /**< forwardFastLeft or forwardFastRight. */
  (float)(2500.f) kickLength,                                /**< Requested kick reach (mm). */
  (float)(.5f) walkSpeed,                                    /**< Walking speed ratio for approach and kick. */
  (MotionRequest::Dive::Request)(MotionRequest::Dive::prepare) diveRequest, /**< Defense for `dive`. */
  (float)(0.f) targetY,                                      /**< Striker aim on the goal line (m, striker's left positive). */
  (float)(0.f) trigger,                                      /**< Policy trigger in [0, 1]. */
  (std::string) policy,                                      /**< Name of the active policy. */
  (std::string) reason,                                      /**< Why this action was chosen. */
});
