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
}
