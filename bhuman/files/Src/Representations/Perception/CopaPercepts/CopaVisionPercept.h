/**
 * @file CopaVisionPercept.h
 *
 * What one camera sees of the Copa penalty scene besides the ball: the blue
 * goal posts, the goal when two posts span a plausible angle, and other robots.
 * Bearings are robot-relative and reliable; positions on the field come from
 * the ground projection, which was measured up to 2x too far, and are only
 * valid when the lower end of the object is in the image.
 */

#pragma once

#include "Math/Eigen.h"
#include "Streaming/AutoStreamable.h"
#include <vector>

STREAMABLE(CopaVisionPercept,
{
  STREAMABLE(Box,
  {,
    (int)(0) left,
    (int)(0) right,
    (int)(0) top,
    (int)(0) bottom,
    (bool)(false) bottomVisible, /**< The lower end is inside the image (not hidden by the border or the own body). */
    (bool)(false) onField,       /**< positionOnField is valid. */
    (Vector2f)(Vector2f::Zero()) positionOnField,
    (float)(0.f) bearing,        /**< Robot-relative direction of the box center (rad, left positive). */
  });

  STREAMABLE(Robot,
  {,
    (Box) box,
    (bool)(false) fallen,
  }),

  (bool)(false) upper,           /**< Percept of the upper camera. */
  (std::vector<Box>) posts,      /**< Blue goal posts, left to right in the image. */
  (bool)(false) goalSeen,        /**< The outermost posts span a plausible goal angle. */
  (int)(-1) goalLeftPost,        /**< Index into posts of the left post as seen from the robot. */
  (int)(-1) goalRightPost,
  (float)(0.f) goalSpan,         /**< Angle between the posts (rad). */
  (float)(0.f) goalDistance,     /**< Distance to the goal line from goalSpan and the goal width (mm). */
  (float)(0.f) goalCenterBearing,
  (std::vector<Robot>) robots,   /**< Other NAOs found by the camera's robot detector (not the ball). */
});
