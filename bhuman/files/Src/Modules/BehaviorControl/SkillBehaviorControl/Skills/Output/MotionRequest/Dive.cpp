/**
 * @file Dive.cpp
 *
 * This file implements the implementation of the Dive skill.
 *
 * @author Arne Hasselbring
 */

#include "Representations/BehaviorControl/Libraries/LibCheck.h"
#include "Representations/BehaviorControl/Skills.h"
#include "Representations/Configuration/BehaviorParameters.h"
#include "Representations/MotionControl/KeyframeMotionRequest.h"
#include "Representations/MotionControl/MotionInfo.h"
#include "Representations/MotionControl/MotionRequest.h"

SKILL_IMPLEMENTATION(DiveImpl,
{,
  IMPLEMENTS(Dive),
  REQUIRES(BehaviorParameters),
  REQUIRES(LibCheck),
  REQUIRES(MotionInfo),
  MODIFIES(MotionRequest),
});

class DiveImpl : public DiveImplBase
{
  /**
   * PenaltyStack: without jumping, the keeper may sit (prepare -> sitDownKeeper) or
   * genuflect (squat with the arms back, B-Human's keeper block), but a side jump
   * becomes the genuflect to the same side. It never throws itself to the floor.
   */
  static MotionRequest::Dive::Request withoutJump(MotionRequest::Dive::Request request)
  {
    if(request == MotionRequest::Dive::jumpLeft)
      return MotionRequest::Dive::squatWideArmsBackLeft;
    if(request == MotionRequest::Dive::jumpRight)
      return MotionRequest::Dive::squatWideArmsBackRight;
    return request;
  }

  void execute(const Dive& p) override
  {
    if(theBehaviorParameters.keeperJumpingOn)
    {
      theMotionRequest.motion = MotionRequest::dive;
      theMotionRequest.diveRequest = p.request;
    }
    else if(theBehaviorParameters.keeperSitWhenNoDive)
    {
      theMotionRequest.motion = MotionRequest::dive;
      theMotionRequest.diveRequest = withoutJump(p.request);
    }
    else
    {
      theMotionRequest.motion = MotionRequest::stand;
      theMotionRequest.standHigh = true;
    }
    theLibCheck.inc(LibCheck::motionRequest);
  }

  bool isDone(const Dive& p) const override
  {
    if(!theBehaviorParameters.keeperJumpingOn)
    {
      if(!theBehaviorParameters.keeperSitWhenNoDive)
        return true;
    }

    const KeyframeMotionRequest keyframeMotionRequest = KeyframeMotionRequest::fromDiveRequest(
      theBehaviorParameters.keeperJumpingOn ? p.request : withoutJump(p.request));
    return theMotionInfo.isKeyframeMotion(keyframeMotionRequest.keyframeMotion, keyframeMotionRequest.mirror);
  }
};

MAKE_SKILL_IMPLEMENTATION(DiveImpl);
