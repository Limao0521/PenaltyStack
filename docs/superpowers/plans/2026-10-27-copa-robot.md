# Copa robot implementation plan

**Goal:** Provide local striker and sit-only keeper scenarios selectable in B-Human DeployDialog for staged testing.

**Architecture:** A testable local state machine accepts timestamped whistle and touch inputs and produces latched start/stop state. Scenario-specific B-Human behavior owns motion and uses robot-relative ball observations, while existing camera streams remain available for wired trials. Build and test before any deployment.

**Constraints:** No GameController or Wi-Fi dependency during competition; manual non-leg abort; a single kick; never request a diving keyframe; timeout and rearm; preserve unrelated checkout changes.

1. Test start, stale/held signals, stop, timeout and rearm using a self-contained state machine.
2. Integrate the state machine into B-Human with scenario-selected role and local whistle, using existing motion and ball representations. Add two scenarios and verify module wiring and no-dive behavior.
3. Build the Nao target and check DeployDialog scenario discovery; validate stream protocol and prepare operator instructions.
4. Report readiness only to the extent supported by tests. Physical detection, timing and staged motion require observation on the actual NAO.
