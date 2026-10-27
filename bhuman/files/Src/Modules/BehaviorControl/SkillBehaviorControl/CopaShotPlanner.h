/**
 * @file CopaShotPlanner.h
 *
 * Chooses the Copa penalty shot from what the striker saw while sweeping its
 * head over the goal. Only bearings (robot-relative angles) are trusted: the
 * projected ground distance of the posts was measured about 2x too long, while
 * the angle between the posts gives the goal distance through the known goal
 * width (150 cm). The robot is placed facing the rival goal, so the shot is a
 * small turn around the ball, never a walk to the side.
 *
 * planShot is the single decision point; an RL policy can replace it later.
 * Header-only and free of B-Human types so it can be tested on the host.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace CopaShot
{
  constexpr float pi = 3.14159265358979f;
  constexpr float deg = pi / 180.f;

  struct Params
  {
    float goalWidth = 1500.f;       /**< mm between the posts. */
    float minGoalSpan = 25.f * deg; /**< Accepted angle between the two posts... */
    float maxGoalSpan = 95.f * deg; /**< ...seen from the striker. */
    float clusterGap = 8.f * deg;   /**< Bearings closer than this belong to the same post. */
    float postMargin = 0.25f;       /**< Aim this fraction of the goal width inside a post. */
    float maxAim = 20.f * deg;      /**< Largest kick direction (turn around the ball). */
    int defaultSide = 1;            /**< Without keeper: 1 left, -1 right, 0 center. */
  };

  /** Bearings collected during the sweep. */
  struct Observations
  {
    std::vector<float> posts;
    std::vector<float> robots;
  };

  struct Goal
  {
    bool hasLeft = false, hasRight = false;
    float left = 0.f, right = 0.f;  /**< Post bearings (left > right). */
    bool inferred = false;          /**< One post was inferred from the expected span. */
  };

  struct Shot
  {
    float direction = 0.f;          /**< Robot-relative kick direction from the ball. */
    float targetX = 0.f, targetY = 0.f; /**< Aim point on the goal line, robot-relative (mm). */
    int side = 0;                   /**< 1 left, -1 right, 0 center/straight. */
    bool keeperSeen = false;
    float keeper = 0.f;
    bool goalSeen = false;
    float goalLeftY = 0.f, goalRightY = 0.f; /**< Posts on the goal line x = targetX (mm). */
    bool goalInferred = false;
    float keeperY = 0.f;                     /**< Keeper on the goal line (mm), if keeperSeen. */
    const char* reason = "no goal";
  };

  /** Sole outline relative to its middle point (RobotDimensions), and the soles' y offset. */
  struct Feet
  {
    float hip = 50.f, front = 110.f, back = 50.f, inner = 38.77f, outer = 50.02f;
  };

  /** Distance from a box [x0,x1] x [y0,y1] to a point. */
  inline float boxDistance(float x0, float x1, float y0, float y1, float px, float py)
  {
    const float dx = std::max({x0 - px, 0.f, px - x1});
    const float dy = std::max({y0 - py, 0.f, py - y1});
    return std::hypot(dx, dy);
  }

  /**
   * Gap (mm) between the ball's surface and the nearest sole edge, with the robot
   * standing (soles under the hips). Negative: the ball overlaps a foot.
   */
  inline float footClearance(float ballX, float ballY, float ballRadius, const Feet& f)
  {
    const float left = boxDistance(-f.back, f.front, f.hip - f.inner, f.hip + f.outer, ballX, ballY);
    const float right = boxDistance(-f.back, f.front, -f.hip - f.outer, -f.hip + f.inner, ballX, ballY);
    return std::min(left, right) - ballRadius;
  }

  inline float median(std::vector<float> v)
  {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
  }

  /** Clusters of bearings separated by more than gap, as medians from right (negative) to left. */
  inline std::vector<float> clusters(std::vector<float> bearings, float gap)
  {
    std::sort(bearings.begin(), bearings.end());
    std::vector<float> medians, current;
    for(float b : bearings)
    {
      if(!current.empty() && b - current.back() > gap)
      {
        medians.push_back(median(current));
        current.clear();
      }
      current.push_back(b);
    }
    if(!current.empty())
      medians.push_back(median(current));
    return medians;
  }

  /** Angle the goal spans when seen from distance d (straight ahead). */
  inline float spanAt(float distance, const Params& p) { return 2.f * std::atan(0.5f * p.goalWidth / distance); }

  inline Goal findGoal(const Observations& seen, const Params& p, float expectedDistance)
  {
    Goal goal;
    const std::vector<float> posts = clusters(seen.posts, p.clusterGap);
    if(posts.size() >= 2)
    {
      const float span = posts.back() - posts.front();
      if(span >= p.minGoalSpan && span <= p.maxGoalSpan)
      {
        goal.hasLeft = goal.hasRight = true;
        goal.left = posts.back();
        goal.right = posts.front();
        return goal;
      }
    }
    if(posts.empty())
      return goal;
    // One usable post: the robot faces the goal, so its sign tells which one it is.
    const float post = std::abs(posts.front()) > std::abs(posts.back()) ? posts.front() : posts.back();
    const float span = spanAt(expectedDistance, p);
    goal.inferred = true;
    if(post >= 0.f)
    {
      goal.hasLeft = true;
      goal.left = post;
      goal.right = post - span;
    }
    else
    {
      goal.hasRight = true;
      goal.right = post;
      goal.left = post + span;
    }
    return goal;
  }

  struct Pose
  {
    float rotation = 0.f, x = 0.f, y = 0.f;
  };

  /**
   * Pose relative to the robot from which B-Human's WalkToBallAndKickEngine starts the
   * kick (kickPose = Pose2f(direction, ball).rotate(rotationOffset).translate(ballOffset)),
   * moved `clearance` mm further back along the kick direction. Aligning there keeps the
   * feet off the ball; the kick then only walks straight forward.
   */
  inline Pose preKickPose(float ballX, float ballY, float direction, float offsetX, float offsetY,
                          float rotationOffset, float clearance)
  {
    Pose pose;
    pose.rotation = direction + rotationOffset;
    const float c = std::cos(pose.rotation), s = std::sin(pose.rotation);
    const float ox = offsetX - clearance, oy = offsetY;
    pose.x = ballX + c * ox - s * oy;
    pose.y = ballY + s * ox + c * oy;
    return pose;
  }

  /**
   * @param ballX, ballY ball position relative to the robot (mm).
   * @param expectedDistance distance to the goal line used when only one post is seen (mm).
   */
  inline Shot planShot(const Observations& seen, float ballX, float ballY, const Params& p, float expectedDistance = 1500.f)
  {
    Shot shot;
    const Goal goal = findGoal(seen, p, expectedDistance);
    if(!goal.hasLeft && !goal.hasRight)
    {
      shot.targetX = expectedDistance;
      shot.direction = 0.f;
      return shot;
    }
    shot.goalSeen = true;
    const float span = goal.left - goal.right;
    const float distance = 0.5f * p.goalWidth / std::tan(0.5f * span) * std::cos(0.5f * (goal.left + goal.right));
    const float leftY = distance * std::tan(goal.left);
    const float rightY = distance * std::tan(goal.right);
    const float margin = p.postMargin * (leftY - rightY);

    // Keeper: the robot between (or close to) the posts that is nearest to the goal center.
    float best = 1e9f;
    for(float b : seen.robots)
      if(b <= goal.left + 5.f * deg && b >= goal.right - 5.f * deg)
      {
        const float off = std::abs(b - 0.5f * (goal.left + goal.right));
        if(off < best)
        {
          best = off;
          shot.keeperSeen = true;
          shot.keeper = b;
        }
      }

    if(shot.keeperSeen)
    {
      const float keeperY = distance * std::tan(shot.keeper);
      shot.side = leftY - keeperY >= keeperY - rightY ? 1 : -1;
      shot.reason = "away from keeper";
    }
    else
    {
      shot.side = p.defaultSide;
      shot.reason = "no keeper, default side";
    }
    shot.goalLeftY = leftY;
    shot.goalRightY = rightY;
    shot.goalInferred = goal.inferred;
    if(shot.keeperSeen)
      shot.keeperY = distance * std::tan(shot.keeper);
    shot.targetX = distance;
    shot.targetY = shot.side > 0 ? leftY - margin : shot.side < 0 ? rightY + margin : 0.5f * (leftY + rightY);
    shot.direction = std::clamp(std::atan2(shot.targetY - ballY, shot.targetX - ballX), -p.maxAim, p.maxAim);
    return shot;
  }
}
