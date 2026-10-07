#include "factories.h"
#include "Modules/MotionControl/KeyframeMotionEngine/KeyframeMotionEngine.h"
NativeModule makeNativeKeyframeMotionEngine()
{ return {std::unique_ptr<Streamable>(new KeyframeMotionEngine), KeyframeMotionEngine::getModuleInfo()}; }
