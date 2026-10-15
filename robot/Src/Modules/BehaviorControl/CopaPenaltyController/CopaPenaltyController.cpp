/**
 * @file CopaPenaltyController.cpp
 *
 * @author PenaltyStack
 */

#include "CopaPenaltyController.h"
#include "Debugging/Annotation.h"
#include "Platform/SystemCall.h"
#include <algorithm>
#include <iostream>

MAKE_MODULE(CopaPenaltyController);

CopaPenaltyController::CopaPenaltyController()
{
  wasPressed.fill(false);
  lastDuration.fill(0);
  mixed.fill(false);
  tapped.fill(false);
  state.role = role;
  state.startMode = startMode;
  state.phase = CopaPenaltyState::unstiff;
}

void CopaPenaltyController::update(CopaPenaltyState& copaPenaltyState)
{
  step();
  copaPenaltyState = state;
}

void CopaPenaltyController::update(GameState& gameState)
{
  step();
  const bool striker = state.role == CopaPenaltyState::striker;
  gameState = GameState(); // team number, colors and player number from the deployed settings
  gameState.phase = GameState::penaltyShootout;
  gameState.ownTeam.goalkeeperNumber = striker ? (gameState.playerNumber == 1 ? 2 : 1) : gameState.playerNumber;
  gameState.ownTeam.playerStates.fill(GameState::active);
  gameState.opponentTeam.playerStates.fill(GameState::active);
  gameState.gameControllerActive = false;
  gameState.timeWhenStateStarted = state.timeWhenPhaseStarted;
  gameState.timeWhenPlayerStateStarted = state.timeWhenPhaseStarted;
  gameState.whistled = state.phase == CopaPenaltyState::playing && state.startSource == CopaPenaltyState::refereeWhistle &&
                       state.timeWhenPhaseStarted == theFrameInfo.time;
  switch(state.phase)
  {
    case CopaPenaltyState::unstiff:
      gameState.state = GameState::beforeHalf;
      gameState.playerState = GameState::unstiff;
      break;
    case CopaPenaltyState::waiting:
      gameState.state = striker ? GameState::waitForOwnPenaltyShot : GameState::waitForOpponentPenaltyShot;
      gameState.playerState = GameState::active;
      break;
    case CopaPenaltyState::playing:
      gameState.state = striker ? GameState::ownPenaltyShot : GameState::opponentPenaltyShot;
      gameState.playerState = GameState::active;
      gameState.timeWhenStateEnds = state.deadline;
      break;
    default:
      gameState.state = striker ? GameState::afterOwnPenaltyShot : GameState::afterOpponentPenaltyShot;
      gameState.playerState = GameState::active;
      break;
  }
}

void CopaPenaltyController::step()
{
  if(stepped && lastStep == theFrameInfo.time)
    return;
  stepped = true;
  lastStep = theFrameInfo.time;
  detectGestures();

  if(unstiffGesture && state.phase != CopaPenaltyState::unstiff)
  {
    say("joints off");
    setPhase(CopaPenaltyState::unstiff);
    return;
  }

  switch(state.phase)
  {
    case CopaPenaltyState::unstiff:
      if(tapped[KeyStates::chest])
      {
        setPhase(CopaPenaltyState::waiting);
        say(roleName() + " ready");
      }
      break;

    case CopaPenaltyState::waiting:
    {
      if(roleGesture)
      {
        state.role = state.role == CopaPenaltyState::striker ? CopaPenaltyState::keeper : CopaPenaltyState::striker;
        setPhase(CopaPenaltyState::waiting);
        say(roleName());
        break;
      }
      const bool buttonAllowed = state.startMode != CopaPenaltyState::whistleOnly;
      const bool whistleAllowed = state.startMode != CopaPenaltyState::buttonOnly;
      if(buttonAllowed && tapped[KeyStates::headMiddle])
        start(CopaPenaltyState::headButton);
      else if(whistleAllowed && theWhistle.lastTimeWhistleDetected > lastAcceptedWhistle &&
              theWhistle.lastTimeWhistleDetected > state.timeWhenPhaseStarted + whistleArmDelayMs &&
              theWhistle.confidenceOfLastWhistleDetection >= minWhistleConfidence)
      {
        lastAcceptedWhistle = theWhistle.lastTimeWhistleDetected;
        start(CopaPenaltyState::refereeWhistle);
      }
      break;
    }

    case CopaPenaltyState::playing:
      if(abortGesture)
      {
        setPhase(CopaPenaltyState::aborted);
        say("stop");
      }
      else if(theFrameInfo.time >= state.deadline)
      {
        setPhase(CopaPenaltyState::finished);
        say("time");
      }
      else if(theCopaPenaltyDecision.command == CopaPenaltyDecision::done)
      {
        if(!timeWhenDoneReported)
          timeWhenDoneReported = theFrameInfo.time;
        else if(theFrameInfo.getTimeSince(timeWhenDoneReported) >= static_cast<int>(doneGraceMs))
          setPhase(CopaPenaltyState::finished);
      }
      break;

    default: // finished, aborted: stand still until re-armed for the next shot
      if(tapped[KeyStates::headFront])
      {
        setPhase(CopaPenaltyState::waiting);
        say(roleName() + " ready");
      }
      break;
  }
}

void CopaPenaltyController::detectGestures()
{
  const KeyStates::Key head[] = {KeyStates::headFront, KeyStates::headMiddle, KeyStates::headRear};
  FOREACH_ENUM(KeyStates::Key, key)
  {
    const bool pressed = theEnhancedKeyStates.pressed[key];
    tapped[key] = false;
    if(pressed)
    {
      if(!wasPressed[key])
        mixed[key] = false;
      lastDuration[key] = theEnhancedKeyStates.pressedDuration[key];
      for(const KeyStates::Key other : head)
        if(other != key && theEnhancedKeyStates.pressed[other])
          mixed[key] = true;
    }
    else if(wasPressed[key])
      tapped[key] = !mixed[key] && lastDuration[key] >= tapMinMs && lastDuration[key] <= tapMaxMs;
    wasPressed[key] = pressed;
  }

  const auto heldFor = [this](KeyStates::Key key, unsigned duration) { return theEnhancedKeyStates.isPressedFor(key, duration); };
  const bool allHead = theEnhancedKeyStates.pressed[KeyStates::headFront] && theEnhancedKeyStates.pressed[KeyStates::headMiddle] &&
                       theEnhancedKeyStates.pressed[KeyStates::headRear];
  const unsigned allHeadDuration = std::min({theEnhancedKeyStates.pressedDuration[KeyStates::headFront],
                                             theEnhancedKeyStates.pressedDuration[KeyStates::headMiddle],
                                             theEnhancedKeyStates.pressedDuration[KeyStates::headRear]});
  unstiffGesture = allHead && allHeadDuration >= unstiffHoldMs;
  abortGesture = heldFor(KeyStates::headRear, abortHoldMs);
  roleGesture = heldFor(KeyStates::headFront, roleToggleHoldMs) && !theEnhancedKeyStates.pressed[KeyStates::headMiddle] &&
                !theEnhancedKeyStates.pressed[KeyStates::headRear];
}

void CopaPenaltyController::setPhase(CopaPenaltyState::Phase phase)
{
  if(state.phase != phase)
  {
    ANNOTATION("CopaPenaltyController", TypeRegistry::getEnumName(state.phase) << " -> " << TypeRegistry::getEnumName(phase));
    // bhuman -b writes stdout to /home/nao/logging: one line per phase change for the post-shot review.
    std::cout << "[CopaPenalty] t=" << theFrameInfo.time << " phase " << TypeRegistry::getEnumName(state.phase) << " -> "
              << TypeRegistry::getEnumName(phase) << " role=" << roleName() << std::endl;
  }
  state.phase = phase;
  state.timeWhenPhaseStarted = theFrameInfo.time;
  timeWhenDoneReported = 0;
}

void CopaPenaltyController::start(CopaPenaltyState::StartSource source)
{
  setPhase(CopaPenaltyState::playing);
  state.startSource = source;
  state.timeWhenStarted = theFrameInfo.time;
  state.deadline = theFrameInfo.time + activeDurationMs;
  ++state.attempt;
  ANNOTATION("CopaPenaltyController", "start " << TypeRegistry::getEnumName(source) << " role " << roleName());
  std::cout << "[CopaPenalty] t=" << theFrameInfo.time << " start " << TypeRegistry::getEnumName(source) << " attempt "
            << state.attempt << " role=" << roleName() << std::endl;
}

void CopaPenaltyController::say(const std::string& text) const
{
  if(speak)
    SystemCall::say(text.c_str());
}

std::string CopaPenaltyController::roleName() const
{
  return state.role == CopaPenaltyState::striker ? "striker" : "keeper";
}
