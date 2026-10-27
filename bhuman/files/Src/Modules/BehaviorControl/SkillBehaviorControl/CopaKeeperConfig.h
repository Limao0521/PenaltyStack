/**
 * @file CopaKeeperConfig.h
 *
 * Parameters of the Copa goalkeeper (see CopaKeeperPlanner.h), loaded from
 * copaKeeper.cfg only in the CopaPenaltyKeeper scenario.
 */

#pragma once

#include "Streaming/AutoStreamable.h"

STREAMABLE(CopaKeeperConfig,
{,
  (float)(150.f) minApproachSpeed,  /**< mm/s towards the keeper to count as a shot. */
  (float)(4.f) maxTime,             /**< s; later crossings are ignored. */
  (float)(0.8f) genuflectTime,      /**< s; closer crossings are blocked with the genuflect. */
  (float)(500.f) genuflectDistance, /**< mm; a ball this close is blocked with the genuflect. */
  (float)(80.f) standRadius,        /**< mm; crossing this close to the body: just stand. */
  (float)(700.f) maxLateral,        /**< mm the keeper may walk sideways from its start (goal area). */
  (int)(150) confirmMs,             /**< A threat must last this long. */
  (int)(2500) holdMs,               /**< Keep the genuflect at least this long. */
  (int)(700) seenTimeoutMs,         /**< The ball counts as seen this long after a sighting (3-4 Hz detections). */
  (int)(400) releaseMs,             /**< A shot stays active this long after its last sighting. */
  (float)(40.f) walkStartMargin,    /**< Start walking only beyond standRadius + this (hysteresis). */
  (float)(1.f) walkSpeed,           /**< Sideways walk speed ratio. */
  (int)(600) velocityWindowMs,      /**< Ball sightings used for the keeper's own velocity fit. */
  (int)(0) blockMotion,             /**< When the ball arrives: 0 stand (body block), 1 sit (sitDownKeeper),
                                         2 genuflect (genuflectStandDefender, drops to a split on the floor). */
  (int)(500) telemetryMs,           /**< [CopaKeeperBall] log period while running; 0 disables it. */
});
