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
    // Known placement: robot 200 mm behind the mark, goal line 1300 mm beyond it.
    const PriorGoal prior;
    int matched = -1;
    // Nothing seen: no correction, the known goal is used; default side left, inside the post.
    assert(headingCorrection(Observations{}, prior, 12.f * deg, 15.f * deg, matched) == 0.f && matched == 0);
    const Shot plain = planShotWithPrior(Observations{}, 200.f, 0.f, prior, 0.f, p);
    assert(plain.goalSeen && plain.side == 1 && near(plain.targetX, 1500.f, 1e-2f) && near(plain.targetY, 375.f, 1e-2f));
    assert(plain.direction > 0.f && plain.direction <= p.maxAim);
    // Robot placed 8 deg turned to the right: the posts appear 8 deg further left.
    const float turn = 8.f * deg;
    Observations seen{{left + turn, left + turn + 0.005f, right + turn}, {}};
    const float correction = headingCorrection(seen, prior, 12.f * deg, 15.f * deg, matched);
    assert(matched == 3 && near(correction, turn, 0.01f));
    Params center = p;
    center.defaultSide = 0;
    assert(near(planShotWithPrior(Observations{}, 200.f, 0.f, prior, 0.f, center).direction, 0.f, 1e-4f));
    const Shot corrected = planShotWithPrior(seen, 200.f, 0.f, prior, correction, center);
    assert(corrected.direction > 7.f * deg && corrected.direction < 10.f * deg); // turns left to the goal center
    // Blue things far from the known posts are ignored.
    Observations noise{{left + 30.f * deg}, {}};
    assert(headingCorrection(noise, prior, 12.f * deg, 15.f * deg, matched) == 0.f && matched == 0);
    // Keeper right of center: shoot left; keeper left: shoot right.
    Observations keeperRight{{}, {bearing(1500.f, -200.f)}};
    const Shot left1 = planShotWithPrior(keeperRight, 200.f, 0.f, prior, 0.f, p);
    assert(left1.keeperSeen && left1.side == 1 && near(left1.keeperY, -200.f, 5.f));
    Observations keeperLeft{{}, {bearing(1500.f, 250.f)}};
    assert(planShotWithPrior(keeperLeft, 200.f, 0.f, prior, 0.f, p).side == -1);
  }
  {
    // Range: lower camera 470 mm high at (50, 0); the ball truly 310 mm ahead, seen at 460 mm.
    const float h = 470.f, r = 32.5f;
    const float pitch = CopaRange::pitchOffsetFromKnown(50.f, 0.f, h, r, 460.f, 0.f, 310.f);
    assert(pitch > 0.f);
    float x = 460.f, y = 0.f;
    CopaRange::correctRange(50.f, 0.f, h, r, pitch, x, y);
    assert(near(x, 310.f, 0.5f) && near(y, 0.f, 1e-3f));
    // The same pitch also shortens other overestimated distances, keeping the bearing.
    float x2 = 600.f, y2 = 100.f;
    CopaRange::correctRange(50.f, 0.f, h, r, pitch, x2, y2);
    assert(x2 < 600.f && x2 > 310.f && near(std::atan2(y2, x2 - 50.f), std::atan2(100.f, 550.f), 1e-4f));
    // No offset: nothing changes.
    float x3 = 400.f, y3 = 30.f;
    CopaRange::correctRange(50.f, 0.f, h, r, 0.f, x3, y3);
    assert(near(x3, 400.f, 1e-2f) && near(y3, 30.f, 1e-2f));
  }
  {
    // Kick box: robot 310 mm behind the ball at start; soles 110 mm long, hip 50 + outer 50.
    const KickBox box;
    const float front = 110.f, hip = 50.f, outer = 50.02f, radius = 32.5f;
    Pose start;
    start.x = -310.f;
    const Pose same = clampToKickBox(start, box, front, hip, outer, radius);
    assert(same.x == -310.f && same.y == 0.f);
    // A target past the line through the ball is pulled back behind it.
    Pose past;
    past.x = -120.f;
    const Pose behind = clampToKickBox(past, box, front, hip, outer, radius);
    assert(near(behind.x, -(radius + box.lineMargin + front), 1e-3f));
    // Turned 20 deg the outer toe reaches further forward, so the limit moves back.
    past.rotation = 20.f * deg;
    assert(clampToKickBox(past, box, front, hip, outer, radius).x < behind.x - 20.f);
    // Too far to the side or back, or turned too much: clamped.
    Pose wild;
    wild.x = -900.f;
    wild.y = 400.f;
    wild.rotation = 60.f * deg;
    const Pose tamed = clampToKickBox(wild, box, front, hip, outer, radius);
    assert(tamed.x == -box.maxBack && tamed.y == box.maxLateral && near(tamed.rotation, box.maxTurn, 1e-6f));
    // The pre-kick poses of both feet are inside the box (kick 190 mm + 100 mm behind, +-50 mm).
    for(float side : {-50.f, 50.f})
    {
      Pose pre;
      pre.x = -290.f;
      pre.y = side;
      const Pose kept = clampToKickBox(pre, box, front, hip, outer, radius);
      assert(kept.x == pre.x && kept.y == pre.y);
    }
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
