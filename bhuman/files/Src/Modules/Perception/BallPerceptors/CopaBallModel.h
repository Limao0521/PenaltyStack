#pragma once
#include <string>

inline bool isCopaScenario(const std::string& scenario)
{
  return scenario == "CopaPenaltyStriker" || scenario == "CopaPenaltyKeeper";
}

inline bool useCopaLowerTrionda(const std::string& scenario, bool lower)
{
  return lower && isCopaScenario(scenario);
}

// Copa scenarios pin both cameras to the models measured on the Copa floor, so
// a checkout whose yolo_ball_320.onnx changes does not change the penalty test.
inline std::string copaBallModel(const std::string& scenario, bool lower, const std::string& configured)
{
  if(!isCopaScenario(scenario))
    return configured;
  return lower ? "NeuralNets/BallDetector/trionda_copa_lower.onnx" : "NeuralNets/BallDetector/trionda_copa_upper.onnx";
}
