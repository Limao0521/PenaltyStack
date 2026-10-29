#include "../Src/Modules/BehaviorControl/SkillBehaviorControl/CopaPenaltyState.h"
#include <cassert>

int main()
{
  CopaPenaltyState state;
  // A held start or old whistle cannot start on initialization.
  assert(state.update(1000, 900, true, false, false) == CopaPenaltyState::stopped);
  assert(state.update(1100, 900, true, false, true) == CopaPenaltyState::waiting);
  assert(state.update(1200, 900, false, false, false) == CopaPenaltyState::waiting);
  assert(state.update(1300, 900, true, false, false) == CopaPenaltyState::running);
  assert(state.update(1400, 900, false, true, false) == CopaPenaltyState::stopped);
  assert(state.update(1500, 1500, false, false, false) == CopaPenaltyState::stopped);
  // Explicit rearm, followed by a new whistle, starts another shot.
  assert(state.update(1600, 1500, false, false, true) == CopaPenaltyState::waiting);
  assert(state.update(1700, 1700, false, false, false) == CopaPenaltyState::running);
  assert(state.update(29699, 1700, false, false, false) == CopaPenaltyState::running);
  assert(state.update(29700, 1700, false, false, false) == CopaPenaltyState::stopped);
  assert(state.update(30000, 1700, false, false, true) == CopaPenaltyState::waiting);
  // A delayed older detection and a detection from before rearming are stale.
  assert(state.update(30100, 1600, false, false, false) == CopaPenaltyState::waiting);
  assert(state.update(30200, 25000, false, false, false) == CopaPenaltyState::waiting);
  assert(state.update(30300, 30300, false, false, false) == CopaPenaltyState::running);
  state.finish();
  assert(state.update(30400, 30300, false, false, false) == CopaPenaltyState::stopped);
  assert(state.update(30500, 30300, false, false, true) == CopaPenaltyState::waiting);
  assert(state.update(30600, 30600, true, false, false, true) == CopaPenaltyState::stopped);
  assert(state.update(30700, 30700, true, false, false) == CopaPenaltyState::stopped);

  {
    // Armed: the whistle starts, the button starts; each records its trigger.
    CopaPenaltyState whistle;
    whistle.update(100, 0, false, false, false);
    assert(whistle.update(200, 0, false, false, true) == CopaPenaltyState::waiting);
    assert(whistle.update(300, 0, false, false, false) == CopaPenaltyState::waiting);
    // Audio clock slightly ahead of the camera frame: accepted on the next cycle.
    assert(whistle.update(400, 420, false, false, false) == CopaPenaltyState::waiting);
    assert(whistle.update(440, 420, false, false, false) == CopaPenaltyState::running);
    assert(whistle.startedByWhistle() && whistle.shotStart() == 420);
    // A head touch while running does nothing.
    assert(whistle.update(500, 420, true, false, false) == CopaPenaltyState::running);
    assert(whistle.update(600, 420, false, true, false) == CopaPenaltyState::stopped);
    // Whistle while stopped (not armed) does not start, nor later after arming.
    assert(whistle.update(700, 700, false, false, false) == CopaPenaltyState::stopped);
    assert(whistle.update(800, 700, false, false, true) == CopaPenaltyState::waiting);
    assert(whistle.update(900, 700, false, false, false) == CopaPenaltyState::waiting);
    assert(whistle.update(1000, 700, true, false, false) == CopaPenaltyState::running);
    assert(!whistle.startedByWhistle() && whistle.shotStart() == 1000);
  }

  CopaKeeperPolicy keeper;
  assert(!keeper.update(false, -500.f)); // stale perception must not trigger
  assert(!keeper.update(true, 0.f));
  assert(keeper.update(true, -121.f));
  assert(keeper.update(false, 0.f)); // hold the sit when the ball disappears
  keeper.reset();
  assert(!keeper.update(true, 0.f));

  CopaHeadTouchGate gate;
  assert(!gate.start(false, false, false));
  assert(!gate.start(true, false, false));
  assert(!gate.start(true, true, false));
  assert(!gate.start(true, true, true));
  assert(!gate.start(false, false, false)); // three-head shutdown is not a start
  assert(!gate.start(true, false, false));
  assert(gate.start(false, false, false)); // solitary touch is accepted on release

  CopaShutdownLatch shutdown;
  assert(!shutdown.update(false, false));
  assert(shutdown.update(true, false));
  assert(shutdown.update(false, false)); // release must not cause standing
  assert(shutdown.update(false, true));  // B-Human reached unstiff
  assert(!shutdown.update(false, false)); // an explicit wake makes it active again
}
