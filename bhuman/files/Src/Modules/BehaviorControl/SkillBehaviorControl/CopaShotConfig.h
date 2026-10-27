/**
 * @file CopaShotConfig.h
 *
 * Parameters of the Copa striker sequence (find ball, look at goal, kick),
 * loaded from copaShot.cfg only in the CopaPenaltyStriker scenario.
 */

#pragma once

#include "Math/Angle.h"
#include "Streaming/AutoStreamable.h"

STREAMABLE(CopaShotConfig,
{,
  (float)(400.f) approachDistance,   /**< Ball closer than this (mm): stop and look at the goal. */
  (float)(300.f) standOffDistance,   /**< When approaching, stop this far behind the ball (mm). */
  (Angle)(5_deg) headTilt,           /**< Head pitch while looking at the goal (positive = down). */
  (Angle)(25_deg) sweepPan,          /**< Head pans to +sweepPan, 0, -sweepPan. */
  (int)(1200) sweepStepMs,           /**< Time per head position. */
  (int)(500) settleMs,               /**< Ignore images while the head is still moving. */
  (float)(1500.f) goalWidth,         /**< Copa goal: 150 cm between posts. */
  (float)(1500.f) expectedGoalDistance, /**< Striker to goal line when only one post is seen (mm). */
  (Angle)(25_deg) minGoalSpan,
  (Angle)(95_deg) maxGoalSpan,
  (float)(0.25f) postMargin,         /**< Aim this fraction of the goal width inside the chosen post. */
  (Angle)(20_deg) maxAim,            /**< Largest kick direction; the robot only turns around the ball. */
  (int)(1) defaultSide,              /**< Without keeper: 1 left, -1 right, 0 center. */
  (float)(100.f) preKickClearance,   /**< Align this far behind B-Human's kick pose, so the feet stay off the ball. */
  (float)(30.f) alignTolerance,      /**< Position error (mm) at which the kick may start. */
  (Angle)(5_deg) alignAngleTolerance,
  (int)(8000) alignTimeoutMs,        /**< Kick anyway after aligning this long. */
  (float)(40.f) minFootClearance,    /**< Never move forward while the ball is closer than this to a sole (mm). */
});
