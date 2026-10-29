#pragma once

// Local shot lifecycle. Inputs must be sampled every cycle. Once armed, either a
// new whistle (heard after arming) or a front head touch starts the shot; a new
// start press is accepted only after release; abort and timeout latch until rearmed.
class CopaPenaltyState
{
public:
  enum State { waiting, running, stopped };
  void finish() { current = stopped; }
  bool startedByWhistle() const { return byWhistle; }
  unsigned shotStart() const { return startTime; }

  State update(unsigned now, unsigned whistle, bool startPressed, bool abortPressed, bool rearmPressed, bool shutdown = false)
  {
    if(!initialized)
    {
      initialized = true;
      lastWhistle = whistle;
      armedTime = now;
      startWasPressed = startPressed;
      rearmWasPressed = rearmPressed;
      return current;
    }
    const bool startEdge = startPressed && !startWasPressed;
    const bool rearmEdge = rearmPressed && !rearmWasPressed;
    startWasPressed = startPressed;
    rearmWasPressed = rearmPressed;
    const bool newWhistle = whistle > lastWhistle && whistle >= armedTime && whistle <= now;
    if(whistle > lastWhistle && whistle <= now)
      lastWhistle = whistle;
    if(abortPressed || shutdown)
      current = stopped;
    else if(current == stopped && rearmEdge)
    {
      current = waiting;
      armedTime = now;
    }
    else if(current == waiting && (startEdge || newWhistle))
    {
      current = running;
      byWhistle = newWhistle;
      startTime = newWhistle ? whistle : now;
    }
    if(current == running && now - startTime >= 28000u)
      current = stopped;
    return current;
  }

private:
  State current = stopped;
  unsigned startTime = 0;
  unsigned armedTime = 0;
  unsigned lastWhistle = 0;
  bool byWhistle = false;
  bool initialized = false;
  bool startWasPressed = false;
  bool rearmWasPressed = false;
};

// Latches the safe sit decision for the entire shot, including loss of vision.
class CopaKeeperPolicy
{
public:
  bool update(bool ballFresh, float forwardVelocity)
  {
    if(ballFresh && forwardVelocity < -120.f)
      sitting = true;
    return sitting;
  }
  void reset() { sitting = false; }
  bool isSitting() const { return sitting; }
private:
  bool sitting = false;
};

// Keep requesting play-dead after the three-head gesture until B-Human has
// entered unstiff and the operator explicitly wakes the robot again.
class CopaShutdownLatch
{
public:
  bool update(bool threeHeads, bool unstiff)
  {
    if(threeHeads)
      latched = true;
    if(latched && unstiff)
      sawUnstiff = true;
    if(latched && sawUnstiff && !unstiff && !threeHeads)
    {
      latched = false;
      sawUnstiff = false;
    }
    return latched;
  }
private:
  bool latched = false;
  bool sawUnstiff = false;
};

class CopaHeadTouchGate
{
public:
  bool start(bool front, bool middle, bool rear)
  {
    if(front && !frontHeld)
      blocked = middle || rear;
    if(front && (middle || rear))
      blocked = true;
    const bool accepted = frontHeld && !front && !blocked && !middle && !rear;
    frontHeld = front;
    return accepted;
  }
private:
  bool frontHeld = false;
  bool blocked = false;
};
