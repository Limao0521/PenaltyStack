/**
 * @file CameraStreamer.h
 *
 * Declares a module that streams JPEG camera frames over TCP while BHuman runs.
 * Upper camera → port upperPort (default 7777)
 * Lower camera → port lowerPort (default 7778)
 *
 * Wire protocol per frame (see CamfStreamServer):
 *   [4 bytes] magic  "CAMF"
 *   [4 bytes] uint32 little-endian JPEG size
 *   [N bytes] raw JPEG data (standard YCbCr JPEG, displayable by any viewer)
 *   ball metadata, ball spots and raw ball patch (CamfStreamServer::appendBallPayload)
 *
 * All socket calls are non-blocking; the module never delays the perception thread.
 */

#pragma once

#include "CamfStreamServer.h"
#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Infrastructure/CameraStream.h"
#include "Representations/Perception/BallPercepts/BallPercept.h"
#include "Representations/Perception/BallPercepts/BallSpots.h"
#include "Representations/Perception/BallPercepts/RawBallPatch.h"
#include "Framework/Module.h"
#include <vector>

MODULE(CameraStreamer,
{,
  REQUIRES(CameraImage),
  REQUIRES(CameraInfo),
  REQUIRES(BallPercept),
  REQUIRES(BallSpots),
  REQUIRES(RawBallPatch),
  PROVIDES(CameraStream),
  LOADS_PARAMETERS(
  {,
    (int)(7777) upperPort, /**< TCP port for the upper camera stream. */
    (int)(7778) lowerPort, /**< TCP port for the lower camera stream. */
    (bool)(true) enabled,  /**< Set to false to disable streaming entirely. */
  }),
});

class CameraStreamer : public CameraStreamerBase
{
  CamfStreamServer server;
  std::vector<unsigned char> payload;

  void update(CameraStream& cameraStream) override;
};
