/**
 * @file CopaTwinShared.h
 *
 * Digital-twin snapshot of the Copa penalty behavior, written by
 * SkillBehaviorControl (Cognition) and sent by CopaCameraStreamer (Upper).
 * All positions are in the "start frame": the robot's odometry pose when the
 * operator started the run (x forward, y left, mm, rad), so the PC can draw
 * the whole run on one map. Same shared-snapshot pattern as CopaVisionShared,
 * so no thread configuration outside the Copa scenarios changes.
 */

#pragma once

#include <cstdint>
#include <mutex>

struct CopaTwinSnapshot
{
  struct Point { bool valid = false; float x = 0.f, y = 0.f; };
  struct Pose { bool valid = false; float x = 0.f, y = 0.f, rotation = 0.f; };

  unsigned time = 0;        /**< Cognition FrameInfo time (ms). */
  uint32_t run = 0;         /**< Increments each time the operator starts the routine. */
  uint8_t role = 0;         /**< 0 striker, 1 keeper. */
  uint8_t state = 0;        /**< CopaPenaltyState: 0 waiting (armed), 1 running, 2 stopped. */
  uint8_t phase = 0;        /**< Striker: 0 findBall, 1 lookGoal, 2 align, 3 kick. Keeper: action. */
  Pose robot;               /**< Robot pose (always valid while publishing). */
  Point ball;               /**< Ball model estimate. */
  Point mark;               /**< Ball spot memorised on the mark. */
  Point leftPost, rightPost;
  bool goalInferred = false;    /**< Goal only from the placement rule, no post seen yet. */
  Point keeper;             /**< Keeper on the goal line. */
  Point target;             /**< Aim point on the goal line. */
  int8_t side = 0;          /**< 1 left, -1 right, 0 center/straight. */
  Pose preKick;             /**< Pose the striker aligns at before the kick. */
  Pose walkTarget;          /**< Where the behavior is walking to right now. */
  float footClearance = 0.f;    /**< Ball surface to nearest sole edge now (mm). */
  float minFootClearance = 0.f; /**< Smallest clearance before the kick in this run (mm). */
  float ballFromMark = 0.f;     /**< How far the ball moved from the mark (mm). */
  float headingCorrection = 0.f;/**< Placement heading error found from the posts (rad). */
  uint8_t postsMatched = 0;     /**< Post sightings that matched the known goal. */
};

class CopaTwinShared
{
public:
  static CopaTwinShared& instance()
  {
    static CopaTwinShared shared;
    return shared;
  }

  void publish(const CopaTwinSnapshot& snapshot)
  {
    std::lock_guard<std::mutex> lock(mutex);
    latest = snapshot;
  }

  CopaTwinSnapshot read() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    return latest;
  }

private:
  mutable std::mutex mutex;
  CopaTwinSnapshot latest;
};
