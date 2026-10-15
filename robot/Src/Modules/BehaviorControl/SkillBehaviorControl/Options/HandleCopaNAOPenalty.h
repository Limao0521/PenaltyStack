/**
 * @file HandleCopaNAOPenalty.h
 *
 * Root behavior of the Copa NAO CCM MX 2026 penalty routine (scenario CopaNAOPenalty).
 * The phase comes from CopaPenaltyController and the tactical decision from
 * CopaPenaltyTactics; this option only executes it with B-Human's own skills.
 *
 * @author PenaltyStack
 */

// PenaltyStack: Copa NAO CCM MX 2026 penalty routine.

option(HandleCopaNAOPenalty)
{
  // The walk-to-ball-and-kick engine uses the routine's own ball track, independent of
  // field-line localization (the competition floor is wood with black lines).
  const auto useTrackedBall = [&]
  {
    if(!theCopaPenaltyDecision.ballValid)
      return;
    theMotionRequest.ballEstimate.position = theCopaPenaltyDecision.ballPosition;
    theMotionRequest.ballEstimate.velocity = theCopaPenaltyDecision.ballVelocity;
    theMotionRequest.ballEstimateTimestamp = theFrameInfo.time;
    theMotionRequest.ballTimeWhenLastSeen = theCopaPenaltyDecision.ballTimeWhenLastSeen;
  };

  const auto lookAtBall = [&]
  {
    if(theCopaPenaltyDecision.ballValid)
      theLookAtPointSkill({.target = Vector3f(theCopaPenaltyDecision.ballPosition.x(),
                                              theCopaPenaltyDecision.ballPosition.y(),
                                              theBallSpecification.radius)});
    else
      theLookForwardSkill();
  };

  common_transition
  {
    switch(theCopaPenaltyState.phase)
    {
      case CopaPenaltyState::unstiff:
        goto unstiff;
      case CopaPenaltyState::waiting:
        goto waiting;
      case CopaPenaltyState::playing:
        goto playing;
      default:
        goto stopped;
    }
  }

  initial_state(selectPhase)
  {}

  state(unstiff)
  {
    action
    {
      theLookAtAnglesSkill({.pan = JointAngles::off,
                            .tilt = JointAngles::off});
      thePlayDeadSkill();
    }
  }

  // Nothing moves before the start signal (rule 3.3.5): stand still, head straight.
  state(waiting)
  {
    action
    {
      theLookForwardSkill();
      theStandSkill();
    }
  }

  // Finished or deactivated: stay in the execution area until the judge allows removal.
  state(stopped)
  {
    action
    {
      theLookForwardSkill();
      theStandSkill();
    }
  }

  state(playing)
  {
    action
    {
      const float speed = theCopaPenaltyDecision.walkSpeed;
      switch(theCopaPenaltyDecision.command)
      {
        case CopaPenaltyDecision::search:
          theLookLeftAndRightSkill();
          theStandSkill();
          break;
        case CopaPenaltyDecision::approach:
          theWalkToPoseSkill({.target = theCopaPenaltyDecision.approachPose,
                              .speed = Pose2f(speed, speed, speed)});
          lookAtBall();
          break;
        case CopaPenaltyDecision::kick:
          theWalkToBallAndKickSkill({.targetDirection = theCopaPenaltyDecision.kickDirection,
                                     .kickType = theCopaPenaltyDecision.kickType,
                                     .alignPrecisely = KickPrecision::precise,
                                     .kickLength = theCopaPenaltyDecision.kickLength,
                                     .speed = Pose2f(speed, speed, speed),
                                     .preStepAllowed = true,
                                     .turnKickAllowed = false});
          useTrackedBall();
          lookAtBall();
          break;
        case CopaPenaltyDecision::dive:
          theDiveSkill({.request = theCopaPenaltyDecision.diveRequest});
          lookAtBall();
          break;
        default: // guard, recover, done, idle: stand and watch, never follow the ball
          theStandSkill();
          lookAtBall();
          break;
      }
    }
  }
}
