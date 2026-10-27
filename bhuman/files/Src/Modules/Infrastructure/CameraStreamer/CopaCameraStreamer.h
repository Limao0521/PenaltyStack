/**
 * @file CopaCameraStreamer.h
 *
 * CameraStreamer for the Copa penalty scenarios. Sends the same CAMF frame as
 * CameraStreamer (same ports and cameraStreamer.cfg, so DeployDialog's camera
 * switch keeps working) followed by a Copa section with what the robot sees:
 *
 *   [4 bytes] magic "COPA"
 *   uint8 version (1)
 *   uint8 ballOnField | float32 x | float32 y           (BallPercept::positionOnField, mm)
 *   uint8 nPosts  | per post:  Box
 *   uint8 goalSeen | int8 leftPost | int8 rightPost | float32 width | float32 cx | float32 cy
 *   uint8 nRobots | per robot: Box | uint8 fallen
 *   Box = int16 left | int16 right | int16 top | int16 bottom
 *         | uint8 flags (1 = bottomVisible, 2 = onField) | float32 x | float32 y
 *
 * All values little-endian. Readers that only know CAMF skip the section by
 * searching for the next "CAMF" magic.
 */

#pragma once

#include "CamfStreamServer.h"
#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Infrastructure/CameraStream.h"
#include "Representations/Perception/BallPercepts/BallPercept.h"
#include "Representations/Perception/BallPercepts/BallSpots.h"
#include "Representations/Perception/BallPercepts/RawBallPatch.h"
#include "Representations/Perception/CopaPercepts/CopaVisionPercept.h"
#include "Framework/Module.h"
#include <vector>

MODULE(CopaCameraStreamer,
{,
  REQUIRES(CameraImage),
  REQUIRES(CameraInfo),
  REQUIRES(BallPercept),
  REQUIRES(BallSpots),
  REQUIRES(CopaVisionPercept),
  REQUIRES(RawBallPatch),
  PROVIDES(CameraStream),
  LOADS_PARAMETERS(
  {,
    (int)(7777) upperPort, /**< TCP port for the upper camera stream. */
    (int)(7778) lowerPort, /**< TCP port for the lower camera stream. */
    (bool)(true) enabled,  /**< Set to false to disable streaming entirely. */
  }),
});

class CopaCameraStreamer : public CopaCameraStreamerBase
{
public:
  CopaCameraStreamer() : CopaCameraStreamerBase("cameraStreamer.cfg") {}

private:
  CamfStreamServer server;
  std::vector<unsigned char> payload;

  void update(CameraStream& cameraStream) override;
  void appendCopaPayload();
};
