/**
 * @file CopaKeeperPlanner.h
 *
 * Copa goalkeeper decision, from robot-relative ball prediction only (the
 * field localization is not valid on the Copa floor):
 *  - no threat: stand and track the ball with the head;
 *  - ball rolling towards the keeper's line, still far in time: walk sideways
 *    to where it will cross (inside the goal area);
 *  - ball about to arrive and not centered: B-Human genuflect (squat with the
 *    arms back, SabanaHerons' RoboCup keeper block) to that side. Never a dive.
 * A threat must persist confirmMs, so velocity noise at start does not trigger.
 *
 * Header-only and free of B-Human types so it can be tested on the host.
 */

#pragma once

#include <algorithm>
#include <cmath>

namespace CopaKeeper
{
  struct Params
  {
    float minApproachSpeed = 150.f; /**< mm/s towards the keeper to count as a shot. */
    float maxTime = 4.f;            /**< s; later crossings are ignored. */
    float genuflectTime = 0.8f;     /**< s; closer crossings are blocked with the genuflect. */
    float genuflectDistance = 500.f;/**< mm; a ball this close is blocked with the genuflect. */
    float standRadius = 80.f;       /**< mm; crossing this close to the body: just stand. */
    float maxLateral = 700.f;       /**< mm the keeper may walk sideways from its start. */
    int confirmMs = 150;            /**< A threat must last this long. */
    int holdMs = 2500;              /**< Keep the genuflect at least this long. */
    int releaseMs = 400;            /**< A threat stays active this long after its last sighting
                                         (YOLO gives 3-4 detections per second). */
    float walkStartMargin = 40.f;   /**< Start walking only beyond standRadius + this (hysteresis). */
  };

  struct Ball
  {
    bool seen = false;          /**< Seen in the last 300 ms. */
    float x = 0.f, y = 0.f;     /**< Position relative to the keeper (mm). */
    float vx = 0.f;             /**< Velocity, x towards the keeper is negative (mm/s). */
    float crossY = 0.f;         /**< Where it crosses the keeper's y axis (mm, left positive). */
    float crossTime = 1e9f;     /**< When (s). */
  };

  /**
   * Ball velocity from the keeper's own recent sightings, in a frame fixed to the
   * floor (odometry), so the keeper's sideways steps are not taken as ball motion.
   * B-Human's BallModel velocity was not enough to detect hand-rolled shots with
   * the 3-4 Hz YOLO detections.
   */
  class BallTrack
  {
  public:
    void reset() { count = 0; }

    void add(int timeMs, float x, float y)
    {
      if(count > 0 && samples[(next + capacity - 1) % capacity].t == timeMs)
        return;
      samples[next] = {timeMs, x, y};
      next = (next + 1) % capacity;
      count = std::min(count + 1, capacity);
    }

    /** Least-squares velocity (mm/s) over the samples of the last windowMs; false if too few. */
    bool velocity(int now, int windowMs, float& vx, float& vy) const
    {
      float st = 0.f, sx = 0.f, sy = 0.f, stt = 0.f, stx = 0.f, sty = 0.f;
      int n = 0, first = now, last = now - windowMs;
      for(int i = 0; i < count; ++i)
      {
        const Sample& sample = samples[i];
        if(now - sample.t > windowMs)
          continue;
        const float t = 0.001f * static_cast<float>(sample.t - now);
        st += t; sx += sample.x; sy += sample.y; stt += t * t; stx += t * sample.x; sty += t * sample.y;
        first = std::min(first, sample.t);
        last = std::max(last, sample.t);
        ++n;
      }
      const float denominator = n * stt - st * st;
      if(n < 3 || last - first < 200 || denominator <= 1e-6f)
        return false;
      vx = (n * stx - st * sx) / denominator;
      vy = (n * sty - st * sy) / denominator;
      return true;
    }

  private:
    struct Sample { int t; float x, y; };
    static constexpr int capacity = 16;
    Sample samples[capacity] = {};
    int next = 0, count = 0;
  };

  /** Where and when a ball at (x, y) moving with (vx, vy) crosses the keeper's y axis (no friction). */
  inline bool crossing(float x, float y, float vx, float vy, float& crossY, float& crossTime)
  {
    if(x <= 0.f || vx >= 0.f)
      return false;
    crossTime = -x / vx;
    crossY = y + vy * crossTime;
    return true;
  }

  enum class Action { watch, walk, genuflectLeft, genuflectRight };

  struct Decision
  {
    Action kind = Action::watch;
    float walkY = 0.f;          /**< Sideways step target relative to the keeper (mm). */
  };

  class Planner
  {
  public:
    void reset()
    {
      threatSince = -1;
      lastThreat = -1;
      genuflectSince = -1;
      genuflectLeft = false;
      walking = false;
    }

    /**
     * @param now time in ms.
     * @param lateral current sideways offset from the start position (mm, left positive).
     */
    Decision update(int now, const Ball& ball, float lateral, const Params& p)
    {
      Decision decision;
      const bool threatNow = threat(ball, p);
      if(threatNow)
      {
        last = ball;
        lastThreat = now;
        lateralAtLast = lateral;
      }

      if(genuflectSince >= 0)
      {
        if(now - genuflectSince < p.holdMs || threatNow)
        {
          decision.kind = genuflectLeft ? Action::genuflectLeft : Action::genuflectRight;
          return decision;
        }
        genuflectSince = -1;
      }

      // Between detections the shot stays active, predicted from its last sighting.
      if(lastThreat < 0 || now - lastThreat > p.releaseMs)
      {
        threatSince = -1;
        walking = false;
        return decision;
      }
      if(threatSince < 0)
        threatSince = lastThreat;
      if(now - threatSince < p.confirmMs)
        return decision;

      const float elapsed = 0.001f * static_cast<float>(now - lastThreat);
      const float crossTime = last.crossTime - elapsed;
      const float distance = std::max(0.f, std::hypot(last.x, last.y) + last.vx * elapsed);
      const float crossY = last.crossY - (lateral - lateralAtLast); // the keeper may have stepped since
      const bool arriving = crossTime <= p.genuflectTime || distance <= p.genuflectDistance;
      if(arriving)
      {
        walking = false;
        if(std::abs(crossY) <= p.standRadius)
          return decision; // the body is already in the way
        genuflectSince = now;
        genuflectLeft = crossY > 0.f;
        decision.kind = genuflectLeft ? Action::genuflectLeft : Action::genuflectRight;
        return decision;
      }

      // Sideways room left inside the goal area on the side the ball crosses.
      const float reachable = std::clamp(crossY, -p.maxLateral - lateral, p.maxLateral - lateral);
      walking = std::abs(reachable) > p.standRadius + (walking ? 0.f : p.walkStartMargin);
      if(walking)
      {
        decision.kind = Action::walk;
        decision.walkY = reachable;
      }
      return decision;
    }

    bool isGenuflecting() const { return genuflectSince >= 0; }

  private:
    static bool threat(const Ball& ball, const Params& p)
    {
      return ball.seen && ball.x > 0.f && ball.vx <= -p.minApproachSpeed
             && ball.crossTime > 0.f && ball.crossTime <= p.maxTime;
    }

    Ball last;
    float lateralAtLast = 0.f;
    int threatSince = -1;
    int lastThreat = -1;
    int genuflectSince = -1;
    bool genuflectLeft = false;
    bool walking = false;
  };
}
