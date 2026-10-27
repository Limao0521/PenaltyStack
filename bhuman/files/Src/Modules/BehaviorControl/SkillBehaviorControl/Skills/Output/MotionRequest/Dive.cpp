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
  void execute(const Dive& p) override
  {
    if(theBehaviorParameters.keeperJumpingOn)
    {
      theMotionRequest.motion = MotionRequest::dive;
      theMotionRequest.diveRequest = p.request;
    }
    else if(theBehaviorParameters.keeperSitWhenNoDive)
    {
      // PenaltyStack: the keeper must not dive, it only sits down.
      // Dive::prepare is mapped to the keyframe motion sitDownKeeper by the KeyframeMotionEngine.
      theMotionRequest.motion = MotionRequest::dive;
      theMotionRequest.diveRequest = MotionRequest::Dive::prepare;
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
      return theMotionInfo.isKeyframeMotion(KeyframeMotionRequest::sitDownKeeper);
    }

    const KeyframeMotionRequest keyframeMotionRequest = KeyframeMotionRequest::fromDiveRequest(p.request);
    return theMotionInfo.isKeyframeMotion(keyframeMotionRequest.keyframeMotion, keyframeMotionRequest.mirror);
  }
};

MAKE_SKILL_IMPLEMENTATION(DiveImpl);
