/**
 * @file CopaPenaltyState.h
 *
 * Phase of the Copa NAO CCM MX 2026 penalty routine (rules 2.7 and 3.3), provided by
 * CopaPenaltyController. There is no GameController: the robot is armed with the
 * chest button, started by the referee's whistle and/or the head middle button,
 * and stops on its own after the active budget or on manual deactivation.
 *
 * @author PenaltyStack
 */

#pragma once

#include "Streaming/AutoStreamable.h"
#include "Streaming/Enum.h"

STREAMABLE(CopaPenaltyState,
{
  ENUM(Phase,
  {,
    unstiff,  /**< Joints off (boot, or the three head buttons). The chest button arms the robot. */
    waiting,  /**< Standing still and waiting for the start signal. Nothing moves before the whistle. */
    playing,  /**< Routine running; it ends by itself after the active budget. */
    finished, /**< Routine over; standing still until the judge allows removal or the robot is re-armed. */
    aborted,  /**< Manual deactivation (head rear button); standing still until re-armed. */
  });

  ENUM(Role,
  {,
    striker,
    keeper,
  });

  ENUM(StartSource,
  {,
    noStart,
    headButton,
    refereeWhistle,
  });

  ENUM(StartMode,
  {,
    buttonOnly,      /**< Head middle button only (the team activates after the whistle, rule 2.7). */
    whistleOnly,     /**< Referee whistle only. */
    buttonOrWhistle, /**< Whichever comes first. */
  });

  bool isPlaying() const { return phase == playing; }
  bool isStanding() const { return phase != unstiff; },

  (Phase)(unstiff) phase,                /**< Current phase. */
  (Role)(striker) role,                  /**< Current role (deploy default, can be toggled while waiting). */
  (StartMode)(buttonOnly) startMode,         /**< Accepted start signals. */
  (StartSource)(noStart) startSource,       /**< What started the current or last routine. */
  (unsigned)(0) timeWhenPhaseStarted,    /**< Timestamp of the last phase change. */
  (unsigned)(0) timeWhenStarted,         /**< Timestamp when the current or last routine started. */
  (unsigned)(0) deadline,                /**< Timestamp when the active routine must be over. */
  (unsigned)(0) attempt,                 /**< Number of routines started since boot. */
});
