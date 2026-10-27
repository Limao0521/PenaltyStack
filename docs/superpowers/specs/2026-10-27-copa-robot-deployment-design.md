# Copa NAO robot deployment design

## Goal and constraints

Prepare striker and goalkeeper trials on B-Human NAO v6 robots using the existing DeployDialog, then the Copa NAO CCM MX 2026 tournament workflow. Each robot runs locally, with whistle detection and head-touch fallback, a manual stop on a non-leg sensor, and no robot-to-PC dependency during competition. Each shot must stop within the official 30 seconds measured from the whistle. A keeper may sit but must never intentionally request a diving keyframe. The test stream is for wired practice; disconnect it for competition.

## Existing evidence

Ospinao (192.168.49.7) currently runs B-Human in `Carril`, not `PenaltyStack`. Its upper camera provides CAMF on TCP 7777. The local uncommitted `PenaltyStack` scenario enables both CameraStreamer instances and a sit-only Dive mapping, but the behavior uses B-Human game-state penalty shootout and the HSL field model. That is not a complete Copa controller: the Copa pitch is 2.4 m square, has black markings and a blue goal, and its reference penalty mark is 1.3 m from goal. The existing kick and interception logic depend on field pose and HSL dimensions. Existing simulator evidence does not validate these paths on this robot.

## Approaches considered

1. Extend B-Human with two Copa scenarios (striker and goalkeeper), a local shot controller and role-specific behavior using the existing perception, walking, kick and sit keyframe. DeployDialog already selects scenarios. This retains one hardware owner and the camera stream, and is the recommended route.
2. Deploy the separate NAOqi Python package. The robot image has no NAOqi and its physical kick executor is missing, so this cannot meet the request without replacing the platform integration.
3. Use the existing HSL shootout plus GameController. This is unsuitable for the Copa no-wireless, local start workflow and its field model does not match the competition.

## Proposed behavior

- Select `CopaPenaltyStriker` or `CopaPenaltyKeeper` in DeployDialog; no role inferred from player number. Each scenario contains its own settings and enables existing wired camera streams for practice. Turn streaming off for tournament use if the organizer prohibits wired connections during play.
- A local state machine arms safely, ignores stale whistle and held buttons, then starts once on a new local whistle detection or deliberate front-head touch. A separate non-leg manual abort is latched; do not reuse a start press as abort. After a shot or stop, motion stops and explicit rearming is required. A fallback start cannot establish the whistle time: use a conservative active limit and disclose that official timing starts earlier.
- The striker uses real ball perception and robot-relative approach/kick targeting compatible with the small field; one kick maximum. Avoid assuming the HSL field localization or ideal simulated goal bearing is correct. Validate field and ball models against the actual event surface.
- The keeper tracks the ball, stands until an incoming ball is detected, then requests only `sitDownKeeper` and holds that action until the shot ends or abort is pressed. A defense-in-depth guard prevents jump requests even if a different behavior path calls Dive. Test ball observations, stale detections, false triggers, and actual motion requests; sitting has not yet been validated physically on ospinao.
- Whistle recognition must be tested with the robot microphones and competition whistle. Logging must reveal armed, detected start, fallback start, role, abort, timeout, ball detection and requested keyframe. Stream the upper/lower cameras on the existing TCP ports through a compatible viewer; verify observation and motor-control behavior under streaming load.

## Verification gates

Automated checks cover state transitions, stale/held inputs, one kick, stop and timeout, and no-dive requests. Build Nao and verify both scenario configurations and DeployDialog discovery without overwriting existing user changes. Only after these pass, test on a supported robot in stages: observation and stream; start and abort without locomotion; supported slow walking/kick and sit; then incremental free-standing tests with the ball. Verify each selected robot and repeat role/shot rearming. Never claim tournament readiness solely from simulation or compilation.

## Remaining integration decisions

Choose exact arm and abort sensor gestures after checking B-Human system-button conflicts, and prove the local whistle path is actually populated without GameController. Establish how to measure a safe deadline when head-touch fallback follows the referee whistle; if no reliable whistle timestamp is available, it cannot guarantee the official 30-second limit. Confirm head-button activation and live streaming with organizers. The exact motion and visual algorithms must be calibrated against the actual robot, ball, goal and surface.
