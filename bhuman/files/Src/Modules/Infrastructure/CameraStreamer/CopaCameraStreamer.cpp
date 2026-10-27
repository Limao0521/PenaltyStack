/**
 * @file CopaCameraStreamer.cpp
 */

#include "CopaCameraStreamer.h"
#include "Streaming/Output.h"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>

MAKE_MODULE(CopaCameraStreamer);

namespace
{
  void appendBox(std::vector<unsigned char>& out, const CopaVisionPercept::Box& box)
  {
    auto clamp16 = [](int v) { return static_cast<int16_t>(std::clamp(v, -32768, 32767)); };
    CamfStreamServer::append(out, clamp16(box.left));
    CamfStreamServer::append(out, clamp16(box.right));
    CamfStreamServer::append(out, clamp16(box.top));
    CamfStreamServer::append(out, clamp16(box.bottom));
    CamfStreamServer::append(out, static_cast<uint8_t>((box.bottomVisible ? 1 : 0) | (box.onField ? 2 : 0)));
    CamfStreamServer::append(out, box.positionOnField.x());
    CamfStreamServer::append(out, box.positionOnField.y());
  }
}

void CopaCameraStreamer::update(CameraStream& cameraStream)
{
  if(!server.listening() && enabled)
  {
    const int port = (theCameraInfo.camera == CameraInfo::upper) ? upperPort : lowerPort;
    if(server.start(port))
      OUTPUT_TEXT("[CopaCameraStreamer] Listening on port " << port
                  << " (" << (theCameraInfo.camera == CameraInfo::upper ? "upper" : "lower") << ")");
    else
      OUTPUT_TEXT("[CopaCameraStreamer] Failed to bind port " << port << ": " << std::strerror(errno));
  }

  if(!server.listening())
  {
    cameraStream.isStreaming      = false;
    cameraStream.port             = 0;
    cameraStream.connectedClients = 0;
    return;
  }

  server.acceptClients();
  if(server.clientCount() > 0)
  {
    payload.clear();
    CamfStreamServer::appendBallPayload(payload, theBallPercept, theBallSpots, theRawBallPatch);
    appendCopaPayload();
    server.send(theCameraImage, payload);
  }

  cameraStream.isStreaming      = true;
  cameraStream.port             = server.port();
  cameraStream.connectedClients = server.clientCount();
}

void CopaCameraStreamer::appendCopaPayload()
{
  static const unsigned char magic[4] = {'C', 'O', 'P', 'A'};
  payload.insert(payload.end(), magic, magic + 4);
  CamfStreamServer::append(payload, static_cast<uint8_t>(1));

  CamfStreamServer::append(payload, static_cast<uint8_t>(theBallPercept.status == BallPercept::seen ? 1 : 0));
  CamfStreamServer::append(payload, theBallPercept.positionOnField.x());
  CamfStreamServer::append(payload, theBallPercept.positionOnField.y());

  const std::size_t posts = std::min<std::size_t>(theCopaVisionPercept.posts.size(), 20);
  CamfStreamServer::append(payload, static_cast<uint8_t>(posts));
  for(std::size_t i = 0; i < posts; ++i)
    appendBox(payload, theCopaVisionPercept.posts[i]);

  CamfStreamServer::append(payload, static_cast<uint8_t>(theCopaVisionPercept.goalSeen ? 1 : 0));
  CamfStreamServer::append(payload, static_cast<int8_t>(theCopaVisionPercept.goalLeftPost));
  CamfStreamServer::append(payload, static_cast<int8_t>(theCopaVisionPercept.goalRightPost));
  CamfStreamServer::append(payload, theCopaVisionPercept.goalWidth);
  CamfStreamServer::append(payload, theCopaVisionPercept.goalCenterOnField.x());
  CamfStreamServer::append(payload, theCopaVisionPercept.goalCenterOnField.y());

  const std::size_t robots = std::min<std::size_t>(theCopaVisionPercept.robots.size(), 20);
  CamfStreamServer::append(payload, static_cast<uint8_t>(robots));
  for(std::size_t i = 0; i < robots; ++i)
  {
    appendBox(payload, theCopaVisionPercept.robots[i].box);
    CamfStreamServer::append(payload, static_cast<uint8_t>(theCopaVisionPercept.robots[i].fallen ? 1 : 0));
  }
}
