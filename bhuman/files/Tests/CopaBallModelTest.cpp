#include "../Src/Modules/Perception/BallPerceptors/CopaBallModel.h"
#include <cassert>
int main()
{
  const std::string configured = "NeuralNets/BallDetector/current.onnx";
  assert(copaBallModel("CopaPenaltyStriker", true, configured) == "NeuralNets/BallDetector/trionda_copa_lower.onnx");
  assert(copaBallModel("CopaPenaltyKeeper", true, configured) == "NeuralNets/BallDetector/trionda_copa_lower.onnx");
  assert(copaBallModel("CopaPenaltyStriker", false, configured) == "NeuralNets/BallDetector/trionda_copa_upper.onnx");
  assert(copaBallModel("CopaPenaltyKeeper", false, configured) == "NeuralNets/BallDetector/trionda_copa_upper.onnx");
  assert(copaBallModel("4v4_Full", true, configured) == configured);
  assert(copaBallModel("4v4_Full", false, configured) == configured);
}
