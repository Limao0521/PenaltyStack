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
  (float)(200.f) startBallFromToes,  /**< Assumed mark when the ball is not seen: this far in front of the toes. */
  (float)(1300.f) penaltyDistance,   /**< Copa: penalty mark to goal line. */
  (float)(600.f) maxMarkDistance,    /**< The ball seen in front closer than this is taken as the mark. */
  (int)(2000) ballScanMs,            /**< Armed scan: time to find the ball before assuming the mark. */
  (Angle)(25_deg) ballScanTilt,      /**< Armed scan: head pitch to look at the floor in front of the feet. */
  (Angle)(12_deg) postMatchTolerance,/**< A seen post corrects the heading only if this close to a known post. */
  (Angle)(15_deg) maxHeadingCorrection, /**< Largest placement error corrected from the posts. */
  (Angle)(8_deg) maxPitchCorrection, /**< Largest lower-camera pitch offset calibrated from the placement. */
  (float)(450.f) boxMaxBack,         /**< While adjusting: robot origin at most this far behind the ball... */
  (float)(150.f) boxMaxLateral,      /**< ...and this far to either side of it. */
  (float)(0.3f) trackGain,           /**< Real-time twin: share of each ball sighting applied to the mark. */
  (float)(150.f) trackGate,          /**< Sightings further than this from the predicted mark are ignored. */
});
