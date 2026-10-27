#include "../Src/Modules/BehaviorControl/SkillBehaviorControl/CopaKeeperPlanner.h"
#include <cassert>

using namespace CopaKeeper;

namespace
{
  Ball shot(float x, float crossY, float speed)
  {
    Ball ball;
    ball.seen = true;
    ball.x = x;
    ball.y = crossY * 0.5f;
    ball.vx = -speed;
    ball.crossY = crossY;
    ball.crossTime = x / speed;
    return ball;
  }
}

int main()
{
  const Params p;
  {
    // Ball resting on the penalty mark, or with velocity noise: watch only.
    Planner planner;
    Ball resting = shot(1300.f, 0.f, 0.f);
    resting.crossTime = 1e9f;
    for(int t = 0; t < 2000; t += 33)
      assert(planner.update(t, resting, 0.f, p).kind == Action::watch);
    Ball noise = shot(1300.f, 400.f, 400.f);
    assert(planner.update(3000, noise, 0.f, p).kind == Action::watch); // first frame of a threat
    noise.seen = false;
    assert(planner.update(3033, noise, 0.f, p).kind == Action::watch);
  }
  {
    // Slow shot 40 cm to the left: walk there first, then genuflect left when it arrives.
    Planner planner;
    assert(planner.update(0, shot(1300.f, 400.f, 600.f), 0.f, p).kind == Action::watch);
    const Decision walk = planner.update(200, shot(1200.f, 400.f, 600.f), 0.f, p);
    assert(walk.kind == Action::walk && walk.walkY == 400.f);
    assert(planner.update(1800, shot(400.f, 150.f, 600.f), 250.f, p).kind == Action::genuflectLeft);
    // Held while the ball passes, even if it is not seen any more.
    assert(planner.update(3000, Ball(), 250.f, p).kind == Action::genuflectLeft);
    assert(planner.update(4500, Ball(), 250.f, p).kind == Action::watch);
  }
  {
    // Fast shot to the right: no time to walk, genuflect right.
    Planner planner;
    planner.update(0, shot(1300.f, -500.f, 2500.f), 0.f, p);
    assert(planner.update(160, shot(900.f, -500.f, 2500.f), 0.f, p).kind == Action::genuflectRight);
  }
  {
    // Shot at the body: just stand.
    Planner planner;
    planner.update(0, shot(1300.f, 30.f, 2500.f), 0.f, p);
    assert(planner.update(160, shot(900.f, 30.f, 2500.f), 0.f, p).kind == Action::watch);
  }
  {
    // Already 650 mm left of the start: only 50 mm more allowed, so no walk further left.
    Planner planner;
    planner.update(0, shot(1300.f, 600.f, 400.f), 650.f, p);
    const Decision d = planner.update(200, shot(1250.f, 600.f, 400.f), 650.f, p);
    assert(d.kind == Action::watch);
  }
  {
    // Ball moving away (striker dribbling back): never a threat.
    Planner planner;
    Ball away = shot(1300.f, 0.f, 0.f);
    away.vx = 500.f;
    away.crossTime = 1e9f;
    for(int t = 0; t < 1000; t += 33)
      assert(planner.update(t, away, 0.f, p).kind == Action::watch);
  }
}
