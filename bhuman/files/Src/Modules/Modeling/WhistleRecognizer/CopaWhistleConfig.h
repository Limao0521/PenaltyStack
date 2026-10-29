/**
 * @file CopaWhistleConfig.h
 *
 * Extra whistle gate of the Copa penalty scenarios, loaded by WhistleRecognizer
 * from copaWhistle.cfg only there (a file of its own, so the whistleRecognizer.cfg
 * of every other scenario stays valid). A detector opens after a few 5 ms frames
 * of tone, so clicks, squeaks and beeps opened it; a referee blast lasts hundreds
 * of milliseconds.
 */

#pragma once

#include "Streaming/AutoStreamable.h"

STREAMABLE(CopaWhistleConfig,
{,
  (int)(250) minToneMs,     /**< Tone (frames passing all gates) within one detection before it counts as a whistle. */
  (int)(1) minChannels,     /**< Microphones that must hear it at the same time. */
  (bool)(true) logRejected, /**< Log detections that ended shorter than minToneMs (rate-limited by logIntervalMs). */
});
