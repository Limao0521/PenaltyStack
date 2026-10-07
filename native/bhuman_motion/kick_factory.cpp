#include "factories.h"
#include "Modules/MotionControl/KickEngine/KickEngine.h"
NativeModule makeNativeKickEngine()
{ return {std::unique_ptr<Streamable>(new KickEngine), KickEngine::getModuleInfo()}; }
