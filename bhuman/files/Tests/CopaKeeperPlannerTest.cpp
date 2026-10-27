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
  {
    // Ball rolled at 500 mm/s towards the keeper and 100 mm/s to its left, seen every 280 ms.
    BallTrack track;
    for(int t = 0; t <= 840; t += 280)
      track.add(t, 1300.f - 0.5f * t, 0.1f * t);
    float vx = 0.f, vy = 0.f;
    assert(track.velocity(840, 600, vx, vy));
    assert(vx < -495.f && vx > -505.f && vy > 95.f && vy < 105.f);
    float crossY = 0.f, crossTime = 0.f;
    assert(crossing(880.f, 84.f, vx, vy, crossY, crossTime));
    assert(crossTime > 1.7f && crossTime < 1.8f && crossY > 250.f && crossY < 270.f);
    assert(!crossing(880.f, 84.f, 300.f, 0.f, crossY, crossTime)); // moving away
    // Too few or too close sightings: no estimate.
    BallTrack sparse;
    sparse.add(0, 1000.f, 0.f);
    sparse.add(100, 990.f, 0.f);
    assert(!sparse.velocity(100, 600, vx, vy));
    // Same timestamp twice is one sighting.
    sparse.add(100, 990.f, 0.f);
    assert(!sparse.velocity(150, 600, vx, vy));
  }
  {
    // A still ball seen while the keeper steps sideways: in the floor frame it does not move.
    BallTrack track;
    for(int t = 0; t <= 600; t += 150)
      track.add(t, 1300.f, 0.f);
    float vx = 1.f, vy = 1.f;
    assert(track.velocity(600, 600, vx, vy) && vx > -1e-3f && vx < 1e-3f && vy > -1e-3f && vy < 1e-3f);
  }
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
