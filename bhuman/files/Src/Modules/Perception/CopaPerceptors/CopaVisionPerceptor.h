/**
 * @file CopaVisionPerceptor.h
 *
 * Copa penalty perception per camera: blue goal posts (CopaGoalPostDetector),
 * the goal as a pair of posts, and the robots of the existing robot detector
 * projected onto the field. Writes a [CopaVision] summary to the log every
 * few seconds so a test can be checked without a PC.
 */

#pragma once

#include "Framework/Module.h"
#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Perception/CopaPercepts/CopaVisionPercept.h"
#include "Representations/Perception/ImagePreprocessing/BodyContour.h"
#include "Representations/Perception/ImagePreprocessing/CameraMatrix.h"
#include "Representations/Perception/ImagePreprocessing/ImageCoordinateSystem.h"
#include "Representations/Perception/ObstaclesPercepts/ObstaclesImagePercept.h"
#include <chrono>

MODULE(CopaVisionPerceptor,
{,
  REQUIRES(BodyContour),
  REQUIRES(CameraImage),
  REQUIRES(CameraInfo),
  REQUIRES(CameraMatrix),
  REQUIRES(ImageCoordinateSystem),
  REQUIRES(ObstaclesImagePercept),
  PROVIDES(CopaVisionPercept),
  LOADS_PARAMETERS(
  {,
    (int) minBlueCb,          /**< Minimum Cb - 128 of a post pixel. */
    (int) minCbMinusCr,       /**< Minimum Cb - Cr of a post pixel. */
    (float) minRunFraction,   /**< Minimum post height relative to the image height. */
    (float) minAspect,        /**< Minimum post height / width. */
    (float) goalWidth,        /**< Distance between the posts (mm); Copa goal is 1500. */
    (float) goalWidthTolerance, /**< Accepted relative error of the measured goal width. */
    (float) maxDistance,      /**< Farther projections are not trusted (mm). */
    (int) logIntervalMs,      /**< [CopaVision] log period; 0 disables it. */
  }),
});

class CopaVisionPerceptor : public CopaVisionPerceptorBase
{
  void update(CopaVisionPercept& percept) override;

  /** Projects the bottom center of a box onto the field. */
  void project(CopaVisionPercept::Box& box) const;

  void log(const CopaVisionPercept& percept);

  std::chrono::steady_clock::time_point lastLog{};
};
