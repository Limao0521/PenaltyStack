#pragma once
#include "Framework/Module.h"
#include <string>
struct NativeModule
{
  std::unique_ptr<Streamable> instance;
  std::vector<ModuleBase::Info> info;
};
NativeModule makeNativeKickEngine();
NativeModule makeNativeKeyframeMotionEngine();

struct NativeKeeperMotionParameters
{
  float guardTimeScale = 1.f;
  float jumpTimeScale = 1.f;
  float armSpread = 0.f;
  float guardKneeOffset = 0.f;
};

struct JointLimits;
// The original engine still owns interpolation, balance, mirroring and recovery.
// This applies a single bounded skill profile to its copied defensive endpoints.
std::string applyNativeKeeperMotionSkill(NativeModule& module,
    const NativeKeeperMotionParameters& parameters, const JointLimits& limits);
