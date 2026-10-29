/**
 * @file CopaOrangeBallConfig.h
 *
 * Parameters of the orange ball detector (CopaOrangeBall.h), loaded by
 * YoloBallDetector from copaOrangeBall.cfg only in the Copa scenarios. A file
 * of its own instead of LOADS_PARAMETERS keys, so the yoloBallDetector.cfg of
 * every other scenario stays valid.
 */

#pragma once

#include "Streaming/AutoStreamable.h"

STREAMABLE(CopaOrangeBallConfig,
{,
  (int)(2) mode,                  /**< 0 YOLO only (Trionda), 1 orange only (YOLO not loaded), 2 orange first, YOLO when no orange ball. */
  (int)(2) upperStep,             /**< Sampling step of the upper camera (640x480). */
  (int)(2) lowerStep,             /**< Sampling step of the lower camera (320x240). */
  (int)(40) minY,
  (int)(250) maxY,
  (int)(18) minCr,                /**< Minimum Cr - 128 of an orange sample. */
  (int)(-4) maxCb,                /**< Maximum Cb - 128. */
  (int)(50) minCrMinusCb,         /**< Absolute minimum Cr - Cb. */
  (int)(35) floorMargin,          /**< Cr - Cb above the image median (the floor). */
  (float)(0.35f) minHueRatio,     /**< (Cr - 128) / (128 - Cb): yellow below... */
  (float)(1.9f) maxHueRatio,      /**< ...red above. */
  (int)(5) minSamples,
  (float)(0.42f) minFill,
  (float)(1.8f) maxAspect,
  (float)(0.45f) maxRadiusFraction,
  (float)(2.f) sizeTolerance,     /**< Measured / expected radius accepted in [1/t, t]. */
  (float)(0.35f) minConfidence,   /**< Candidates and tracking below are dropped. */
  (float)(0.8f) instantConfidence,/**< A candidate this good starts the track without a second frame. */
  (int)(2) confirmFrames,         /**< Otherwise, frames that must agree before the track starts. */
  (int)(6) maxMissed,             /**< Frames the track survives without a candidate (not published). */
});
