#include "../Src/Modules/BehaviorControl/SkillBehaviorControl/CopaShotPlanner.h"
#include <cassert>
#include <cmath>

using namespace CopaShot;

namespace
{
  bool near(float a, float b, float tolerance) { return std::abs(a - b) <= tolerance; }

  // Bearings of a point (x, y) relative to the robot.
  float bearing(float x, float y) { return std::atan2(y, x); }
}

int main()
{
  const Params p;
  // Striker 20 cm behind the mark: goal line 1.5 m ahead, posts at +-0.75 m. Ball 20 cm ahead.
  const float left = bearing(1500.f, 750.f), right = bearing(1500.f, -750.f);

  {
    // Both posts seen several times with noise; no keeper: default side (left), inside the post.
    Observations seen{{left, left + 0.01f, left - 0.01f, right, right + 0.01f}, {}};
    const Shot shot = planShot(seen, 200.f, 0.f, p);
    assert(shot.goalSeen && !shot.keeperSeen && shot.side == 1);
    assert(near(shot.targetX, 1500.f, 60.f));
    assert(near(shot.targetY, 750.f - 0.25f * 1500.f, 60.f));
    assert(shot.direction > 0.f && shot.direction <= p.maxAim);
  }
  {
    // Keeper standing a bit left of center: shoot right.
    Observations seen{{left, right}, {bearing(1500.f, 150.f)}};
    const Shot shot = planShot(seen, 200.f, 0.f, p);
    assert(shot.keeperSeen && shot.side == -1);
    assert(shot.direction < 0.f && shot.direction >= -p.maxAim);
  }
  {
    // Keeper right of center: shoot left.
    Observations seen{{left, right}, {bearing(1500.f, -200.f)}};
    assert(planShot(seen, 200.f, 0.f, p).side == 1);
  }
  {
    // A robot far outside the goal is not the keeper.
    Observations seen{{left, right}, {bearing(1500.f, 1500.f)}};
    assert(!planShot(seen, 200.f, 0.f, p).keeperSeen);
  }
  {
    // Only the left post: the right one is inferred 1.5 m away.
    Observations seen{{left, left}, {}};
    const Goal goal = findGoal(seen, p, 1500.f);
    assert(goal.inferred && goal.hasLeft && near(goal.right, left - spanAt(1500.f, p), 0.01f));
    const Shot shot = planShot(Observations{{right}, {}}, 200.f, 0.f, p);
    assert(shot.goalSeen && near(shot.targetX, 1500.f, 300.f));
  }
  {
    // Nothing seen: straight ahead.
    const Shot shot = planShot(Observations{}, 200.f, 0.f, p);
    assert(!shot.goalSeen && shot.direction == 0.f);
  }
  {
    // Robot turned 10 deg to the right: the aim compensates, still bounded.
    const float turn = -10.f * deg;
    Observations seen{{left - turn, right - turn}, {}};
    const Shot shot = planShot(seen, 200.f, 0.f, p);
    assert(shot.goalSeen && shot.direction > 10.f * deg - 1e-3f);
    assert(shot.direction <= p.maxAim);
  }
  assert(clusters({0.1f, 0.12f, -0.5f, -0.48f}, 8.f * deg).size() == 2);
  {
    // forwardFastLeftLong: ballOffset (-190, -50), rotationOffset -3 deg.
    // Straight kick, ball 200 mm ahead: pre-kick 100 mm further back than the kick pose.
    const Pose pre = preKickPose(200.f, 0.f, 0.f, -190.f, -50.f, 0.f, 100.f);
    assert(near(pre.x, -90.f, 1e-3f) && near(pre.y, -50.f, 1e-3f) && pre.rotation == 0.f);
    // Aimed 20 deg left: the pose turns around the ball, keeping the same distance to it.
    const float aim = 20.f * deg;
    const Pose turned = preKickPose(200.f, 0.f, aim, -190.f, -50.f, 0.f, 100.f);
    assert(near(turned.rotation, aim, 1e-6f));
    assert(near(std::hypot(turned.x - 200.f, turned.y), std::hypot(290.f, 50.f), 1e-2f));
    assert(turned.y < -50.f); // to the right of the ball, behind it along the kick line
  }
  {
    const Feet feet;
    // Ball on the mark 200 mm ahead, centered: nearest is the inner toe corner (110, 11.23).
    assert(near(footClearance(200.f, 0.f, 32.5f, feet), std::hypot(90.f, 50.f - 38.77f) - 32.5f, 1e-3f));
    // Ball between the feet: the 22 mm gap is narrower than the ball, so it overlaps.
    assert(near(footClearance(50.f, 0.f, 32.5f, feet), (50.f - 38.77f) - 32.5f, 1e-3f));
    assert(footClearance(50.f, 0.f, 32.5f, feet) < 0.f);
    // Touching the left toe.
    assert(footClearance(110.f + 32.5f, 60.f, 32.5f, feet) <= 1e-3f);
    // Goal geometry is reported with the shot.
    const Shot shot = planShot(Observations{{left, right}, {bearing(1500.f, 200.f)}}, 200.f, 0.f, p);
    assert(near(shot.goalLeftY, 750.f, 40.f) && near(shot.goalRightY, -750.f, 40.f) && near(shot.keeperY, 200.f, 20.f));
  }
}
