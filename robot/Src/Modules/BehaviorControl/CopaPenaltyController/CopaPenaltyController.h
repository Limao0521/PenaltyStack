/**
 * @file CopaPenaltyController.h
 *
 * Referee-free control of the Copa NAO CCM MX 2026 penalty routine (rules 2.7 and 3.3).
 * Replaces GameStateProvider in the CopaNAOPenalty scenario: there is no GameController
 * and no network. The phase machine is driven only by the robot's own sensors:
 *
 *  - chest button (tap):            unstiff -> waiting (stands up, nothing else moves)
 *  - head middle button (tap):      waiting -> playing (team activation after the whistle)
 *  - referee whistle (optional):    waiting -> playing
 *  - head rear button (hold):       playing -> aborted (manual deactivation, not on the legs)
 *  - active budget elapsed:         playing -> finished (28 s < 30 s execution limit)
 *  - head front button (tap):       finished/aborted -> waiting (re-armed for the next shot)
 *  - head front button (hold):      toggles striker/keeper while waiting
 *  - three head buttons (hold):     any phase -> unstiff
 *
 * GameState is filled so that the rest of the framework sees a penalty shot:
 * waiting = waitFor*PenaltyShot (whistle listening), playing = *PenaltyShot,
 * finished/aborted = after*PenaltyShot.
 *
 * @author PenaltyStack
 */

#pragma once

#include "Framework/Module.h"
#include "Representations/BehaviorControl/CopaPenaltyDecision.h"
#include "Representations/BehaviorControl/CopaPenaltyState.h"
#include "Representations/Infrastructure/FrameInfo.h"
#include "Representations/Infrastructure/GameState.h"
#include "Representations/Infrastructure/SensorData/KeyStates.h"
#include "Representations/Modeling/Whistle.h"

MODULE(CopaPenaltyController,
{,
  REQUIRES(EnhancedKeyStates),
  REQUIRES(FrameInfo),
  REQUIRES(Whistle),
  USES(CopaPenaltyDecision),
  PROVIDES(CopaPenaltyState),
  PROVIDES(GameState),
  LOADS_PARAMETERS(
  {,
    (CopaPenaltyState::Role) role,                /**< Role after boot (set by the deploy dialog). */
    (CopaPenaltyState::StartMode) startMode,      /**< Accepted start signals (set by the deploy dialog). */
    (unsigned) activeDurationMs,                  /**< Active budget after the start (rules: 30 s to execute). */
    (unsigned) whistleArmDelayMs,                 /**< Whistles heard this soon after arming are ignored. */
    (float) minWhistleConfidence,                 /**< Minimum Whistle::confidenceOfLastWhistleDetection. */
    (unsigned) tapMinMs,                          /**< Shorter presses are bounces. */
    (unsigned) tapMaxMs,                          /**< Longer presses are holds, not taps. */
    (unsigned) abortHoldMs,                       /**< Head rear hold that deactivates the routine. */
    (unsigned) roleToggleHoldMs,                  /**< Head front hold that toggles the role while waiting. */
    (unsigned) unstiffHoldMs,                     /**< Three head buttons hold that switches the joints off. */
    (unsigned) doneGraceMs,                       /**< The striker routine ends this long after the tactics report it is done. */
    (bool) speak,                                 /**< Spoken feedback for every phase change. */
  }),
});

class CopaPenaltyController : public CopaPenaltyControllerBase
{
public:
  CopaPenaltyController();

private:
  void update(CopaPenaltyState& copaPenaltyState) override;
  void update(GameState& gameState) override;

  /** Advances the phase machine once per frame (both representations share it). */
  void step();
  void detectGestures();
  void setPhase(CopaPenaltyState::Phase phase);
  void start(CopaPenaltyState::StartSource source);
  void say(const std::string& text) const;
  std::string roleName() const;

  CopaPenaltyState state;
  unsigned lastStep = 0;
  bool stepped = false;
  unsigned lastAcceptedWhistle = 0;
  unsigned timeWhenDoneReported = 0;

  ENUM_INDEXED_ARRAY(bool, KeyStates::Key) wasPressed;    /**< Pressed in the previous frame. */
  ENUM_INDEXED_ARRAY(unsigned, KeyStates::Key) lastDuration; /**< Duration of the current/last press. */
  ENUM_INDEXED_ARRAY(bool, KeyStates::Key) mixed;          /**< Another head button was pressed during this press. */
  ENUM_INDEXED_ARRAY(bool, KeyStates::Key) tapped;         /**< A clean tap was released this frame. */
  bool unstiffGesture = false;
  bool abortGesture = false;
  bool roleGesture = false;
};
