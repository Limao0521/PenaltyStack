/**
 * @file CopaVisionPercept.h
 *
 * What one camera sees of the Copa penalty scene besides the ball: the blue
 * goal posts, the goal when two posts are a goal width apart, and other robots.
 * Positions on the field are relative to the robot (mm) and only valid when
 * the lower end of the object is in the image.
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
  });

  STREAMABLE(Robot,
  {,
    (Box) box,
    (bool)(false) fallen,
  }),

  (std::vector<Box>) posts,      /**< Blue goal posts, left to right in the image. */
  (bool)(false) goalSeen,        /**< Two posts on the field are a goal width apart. */
  (int)(-1) goalLeftPost,        /**< Index into posts of the left post as seen from the robot. */
  (int)(-1) goalRightPost,
  (float)(0.f) goalWidth,        /**< Measured distance between the posts (mm). */
  (Vector2f)(Vector2f::Zero()) goalCenterOnField,
  (std::vector<Robot>) robots,   /**< Other NAOs found by the camera's robot detector. */
});
