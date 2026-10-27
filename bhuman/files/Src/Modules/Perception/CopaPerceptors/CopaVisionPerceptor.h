/**
 * @file CopaVisionPerceptor.h
 *
 * Copa penalty perception per camera: blue goal posts (CopaGoalPostDetector),
 * the goal as a pair of posts, and the robots of the existing robot detector.
 * Every object gets a robot-relative bearing; the upper camera's bearings are
 * published to CopaVisionShared for the striker behavior. Writes a
 * [CopaVision] summary to the log every few seconds so a test can be checked
 * without a PC.
 */

#pragma once

#include "Framework/Module.h"
#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Infrastructure/FrameInfo.h"
#include "Representations/Perception/BallPercepts/BallPercept.h"
#include "Representations/Perception/CopaPercepts/CopaVisionPercept.h"
#include "Representations/Perception/ImagePreprocessing/BodyContour.h"
#include "Representations/Perception/ImagePreprocessing/CameraMatrix.h"
#include "Representations/Perception/ImagePreprocessing/ImageCoordinateSystem.h"
#include "Representations/Perception/ObstaclesPercepts/ObstaclesImagePercept.h"
#include <chrono>

MODULE(CopaVisionPerceptor,
{,
  REQUIRES(BallPercept),
  REQUIRES(BodyContour),
  REQUIRES(CameraImage),
  REQUIRES(CameraInfo),
  REQUIRES(CameraMatrix),
  REQUIRES(FrameInfo),
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
    (Angle) minGoalSpan,      /**< Accepted angle between the two posts of the goal... */
    (Angle) maxGoalSpan,      /**< ...seen from the robot. */
    (float) maxDistance,      /**< Farther ground projections are not trusted (mm). */
    (float) robotBallOverlap, /**< A robot box this much larger than the ball around the ball is the ball. */
    (int) logIntervalMs,      /**< [CopaVision] log period; 0 disables it. */
  }),
});

class CopaVisionPerceptor : public CopaVisionPerceptorBase
{
  void update(CopaVisionPercept& percept) override;

  /** Projects the bottom center of a box onto the field and computes its bearing. */
  void locate(CopaVisionPercept::Box& box) const;

  /** Robot-relative direction of an image point (rad, left positive). */
  float bearing(const Vector2f& pointInImage) const;

  /** The lower robot detector reports the Trionda as a robot; drop boxes around the ball. */
  bool isBall(const ObstaclesImagePercept::Obstacle& obstacle) const;

  void log(const CopaVisionPercept& percept);

  std::chrono::steady_clock::time_point lastLog{};
};
