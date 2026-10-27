/**
 * @file CopaBallRange.h
 *
 * Range correction for the lower camera's ball in the Copa scenarios. The ground
 * projection (camera height and angle below the horizon) read the ball 100-230 mm too
 * far on ospinao, i.e. the camera's pitch is off. When the striker is armed, the
 * operator guarantees the ball on the mark 20 cm in front of the toes: the behavior
 * calibrates a pitch offset from that known distance, and YoloBallDetector applies it
 * to every lower-camera ball, so the ball model, the kick engine and the twin agree.
 *
 * Header-only; the functions are free of B-Human types for host tests.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>

namespace CopaRange
{
  /**
   * Pitch offset that moves a ball measured at (ballX, ballY) to the known forward distance
   * trueX (from the robot origin) along the same bearing. The range is the camera height over
   * the tangent of the angle below the horizon, as in the ground projection.
   * camX, camY: camera ground point; h: camera height; r: ball radius (mm).
   */
  inline float pitchOffsetFromKnown(float camX, float camY, float h, float r, float ballX, float ballY, float trueX)
  {
    const float dx = ballX - camX, dy = ballY - camY, d = std::hypot(dx, dy);
    if(d < 1.f || dx < 1.f || h <= r)
      return 0.f;
    const float trueDistance = (trueX - camX) * d / dx; // same bearing, known forward distance
    if(trueDistance < 1.f)
      return 0.f;
    return std::atan((h - r) / trueDistance) - std::atan((h - r) / d);
  }

  /** Applies a pitch offset to a ball position measured with the ground projection. */
  inline void correctRange(float camX, float camY, float h, float r, float pitch, float& ballX, float& ballY)
  {
    const float dx = ballX - camX, dy = ballY - camY, d = std::hypot(dx, dy);
    if(d < 1.f || h <= r)
      return;
    const float angle = std::atan((h - r) / d) + pitch;
    if(angle <= 0.02f)
      return;
    const float corrected = (h - r) / std::tan(angle);
    ballX = camX + dx / d * corrected;
    ballY = camY + dy / d * corrected;
  }

  /** Lower-camera pitch offset shared from the behavior (Cognition) to the detector (Lower). */
  inline std::atomic<float>& lowerPitchOffset()
  {
    static std::atomic<float> offset{0.f};
    return offset;
  }
}
