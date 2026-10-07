#pragma once
#include "Framework/Module.h"
struct NativeModule
{
  std::unique_ptr<Streamable> instance;
  std::vector<ModuleBase::Info> info;
};
NativeModule makeNativeKickEngine();
NativeModule makeNativeKeyframeMotionEngine();
