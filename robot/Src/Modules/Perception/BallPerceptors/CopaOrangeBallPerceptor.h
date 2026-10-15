/**
 * @file CopaOrangeBallPerceptor.h
 *
 * Ball perceptor for the Copa NAO CCM MX 2026 penalty field (rules 3.3.4): an orange
 * plastic ball of 65 mm on a wooden floor with black lines. B-Human's ball CNNs and
 * the HSL YOLO detector are trained for other balls and green carpet, so this module
 * segments orange directly in the YUYV image (high V/Cr, low U/Cb), keeps the
 * connected blob whose size matches the projected size of a 65 mm ball, and projects
 * its centre onto the plane at ball height. The robot's own body is masked out.
 *
 * All thresholds are parameters (copaOrangeBallPerceptor.cfg) to be tuned with images
 * from the competition venue.
 *
 * @author PenaltyStack
 */

#pragma once

#include "Framework/Module.h"
#include "Representations/Configuration/BallSpecification.h"
#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Perception/BallPercepts/BallPercept.h"
#include "Representations/Perception/ImagePreprocessing/BodyContour.h"
#include "Representations/Perception/ImagePreprocessing/CameraMatrix.h"
#include "Representations/Perception/MeasurementCovariance.h"
#include "Math/Eigen.h"
#include <vector>

MODULE(CopaOrangeBallPerceptor,
{,
  REQUIRES(BallSpecification),
  REQUIRES(BodyContour),
  REQUIRES(CameraImage),
  REQUIRES(CameraInfo),
  REQUIRES(CameraMatrix),
  REQUIRES(MeasurementCovariance),
  PROVIDES(BallPercept),
  LOADS_PARAMETERS(
  {,
    (int) gridStep,                 /**< Sampling step in full-resolution pixels. */
    (int) minY,                     /**< Darkest accepted luminance. */
    (int) maxY,                     /**< Brightest accepted luminance (specular highlights are rejected). */
    (int) maxU,                     /**< Orange has a low U (Cb). */
    (int) minV,                     /**< Orange has a high V (Cr). */
    (int) minChroma,                /**< Minimum V - U; separates the ball from light wood. */
    (int) minBlobCells,             /**< Smallest blob (in samples) considered a ball. */
    (float) minFillRatio,           /**< Blob samples / bounding-box samples. */
    (float) maxAspectRatio,         /**< Bounding-box elongation. */
    (float) minRadiusRatio,         /**< Measured / expected radius at the projected distance. */
    (float) maxRadiusRatio,
    (float) maxDistance,            /**< mm; farther percepts are ignored (the field is 2.4 m). */
    (Vector2f) rotationDeviation,   /**< Camera rotation uncertainty for the covariance (rad). */
    (bool) useUpperCamera,
    (bool) useLowerCamera,
  }),
});

class CopaOrangeBallPerceptor : public CopaOrangeBallPerceptorBase
{
  void update(BallPercept& ballPercept) override;

  bool isOrange(int x, int y) const;

  std::vector<unsigned char> mask;  /**< 1 = orange sample, 2 = visited. */
  std::vector<int> queue;
};
