#include "json.hpp"
#include "factories.h"
#include "Modules/MotionControl/KeyframeMotionEngine/KeyframeMotionEngine.h"
#undef action

namespace
{
using json = nlohmann::json;

class ParameterizedNativeKeyframeMotionEngine : public KeyframeMotionEngine
{
  const decltype(keyframeBlock) nominalBlocks = static_cast<const decltype(keyframeBlock)&>(keyframeBlock);
  bool applied = false;

public:
  std::string apply(const NativeKeeperMotionParameters& parameters, const JointLimits& limits)
  {
    if(applied) throw std::runtime_error("Keeper motion skill can only be applied once per runtime");
    const auto bounded = [](float value, float low, float high)
    { return std::isfinite(value) && value >= low && value <= high; };
    if(!bounded(parameters.guardTimeScale, .85f, 1.15f) ||
       !bounded(parameters.jumpTimeScale, .85f, 1.15f) ||
       !bounded(parameters.armSpread, -.08f, .12f) ||
       !bounded(parameters.guardKneeOffset, 0.f, .10f))
      throw std::runtime_error("Keeper motion skill parameter outside bounds");

    // Rebuild from the immutable configuration, never from a previous variant.
    // Runtime lifetimes are one trial; get-up blocks, PID, stiffness and branch
    // conditions remain the original B-Human definitions.
    keyframeBlock = nominalBlocks;
    json blocks = json::array();
    unsigned changedFrames = 0, changedAngles = 0, clampedAngles = 0;
    const auto tuneBlock = [&](KeyframeMotionBlockID::KeyframeMotionBlockID blockID,
                               bool guard, bool onlyJumpReach)
    {
      auto& block = keyframeBlock[blockID];
      const auto& nominal = nominalBlocks[blockID];
      json frames = json::array();
      for(std::size_t index = 0; index < block.keyframes.size(); ++index)
      {
        auto& frame = block.keyframes[index];
        const auto& original = nominal.keyframes[index];
        // The jump has a 1 ms branch marker, a 400 ms reach, then 100/1000 ms
        // relaxation and a 1 ms terminal marker. Only its reach is trainable.
        if(original.duration <= 1.f || (onlyJumpReach && index != 1)) continue;
        const float scale = guard ? parameters.guardTimeScale : parameters.jumpTimeScale;
        if(scale != 1.f) frame.duration = original.duration * scale;
        bool changed = frame.duration != original.duration;
        json angles = json::array();
        const auto tuneAngle = [&](KeyframeLimb limb, std::size_t offset,
                                   Joints::Joint joint, float delta)
        {
          const float base = original.angles.positions[limb][offset];
          if(base == JointAngles::off || base == JointAngles::ignore) return;
          float target = base;
          bool clamped = false;
          // Offset zero must preserve even upstream values outside a limit.
          if(delta != 0.f)
          {
            const float proposed = base + delta;
            target = std::clamp(proposed, static_cast<float>(limits.limits[joint].min),
                                        static_cast<float>(limits.limits[joint].max));
            clamped = target != proposed;
            frame.angles.positions[limb][offset] = target;
          }
          if(target != base) { ++changedAngles; changed = true; }
          clampedAngles += clamped;
          angles.push_back({{"joint",TypeRegistry::getEnumName(joint)},
                            {"nominal_rad",base},{"target_rad",target},{"clamped",clamped}});
        };
        tuneAngle(KeyframeLimb::leftArm, 1, Joints::lShoulderRoll, parameters.armSpread);
        tuneAngle(KeyframeLimb::rightArm, 1, Joints::rShoulderRoll, -parameters.armSpread);
        if(guard)
        {
          tuneAngle(KeyframeLimb::leftLeg, 3, Joints::lKneePitch, parameters.guardKneeOffset);
          tuneAngle(KeyframeLimb::rightLeg, 3, Joints::rKneePitch, parameters.guardKneeOffset);
        }
        changedFrames += changed;
        frames.push_back({{"index",index},{"duration_ms_original",original.duration},
                          {"duration_ms_applied",frame.duration},{"angles",angles}});
      }
      blocks.push_back({{"id",TypeRegistry::getEnumName(blockID)},{"keyframes",frames}});
    };
    tuneBlock(KeyframeMotionBlockID::genuflectFromStand, true, false);
    tuneBlock(KeyframeMotionBlockID::genuflectFromSitting, true, false);
    tuneBlock(KeyframeMotionBlockID::genuflect, true, false);
    tuneBlock(KeyframeMotionBlockID::genuflectWide, true, false);
    tuneBlock(KeyframeMotionBlockID::sitKeeperJump, false, false);
    tuneBlock(KeyframeMotionBlockID::jumpLeftFromStand, false, false);
    tuneBlock(KeyframeMotionBlockID::jumpLeftFromSitting, false, true);
    applied = true;
    return json({{"version","keeper_motion_skill_v1"},{"applied",true},
                 {"parameters",{{"guard_time_scale",parameters.guardTimeScale},
                                {"jump_time_scale",parameters.jumpTimeScale},
                                {"arm_spread_rad",parameters.armSpread},
                                {"guard_knee_offset_rad",parameters.guardKneeOffset}}},
                 {"changed_keyframes",changedFrames},{"changed_angles",changedAngles},
                 {"clamped_angles",clampedAngles},{"blocks",blocks}}).dump();
  }
};
}

NativeModule makeNativeKeyframeMotionEngine()
{ return {std::unique_ptr<Streamable>(new ParameterizedNativeKeyframeMotionEngine), KeyframeMotionEngine::getModuleInfo()}; }

std::string applyNativeKeeperMotionSkill(NativeModule& module,
    const NativeKeeperMotionParameters& parameters, const JointLimits& limits)
{
  auto* engine = dynamic_cast<ParameterizedNativeKeyframeMotionEngine*>(module.instance.get());
  if(!engine) throw std::runtime_error("Keeper motion skill requires native keyframe engine");
  return engine->apply(parameters, limits);
}
