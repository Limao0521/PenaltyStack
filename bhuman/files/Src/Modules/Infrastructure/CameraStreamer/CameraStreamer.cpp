/**
 * @file CameraStreamer.cpp
 *
 * Implementation of the CameraStreamer module.
 * Streams proper YCbCr JPEG frames over a non-blocking TCP server.
 */

#include "CameraStreamer.h"
#include "Streaming/Output.h"
#include <cerrno>
#include <cstring>

MAKE_MODULE(CameraStreamer);

void CameraStreamer::update(CameraStream& cameraStream)
{
  // Lazy init: pick port based on which camera thread we are in
  if(!server.listening() && enabled)
  {
    const int port = (theCameraInfo.camera == CameraInfo::upper) ? upperPort : lowerPort;
    if(server.start(port))
      OUTPUT_TEXT("[CameraStreamer] Listening on port " << port
                  << " (" << (theCameraInfo.camera == CameraInfo::upper ? "upper" : "lower") << ")");
    else
      OUTPUT_TEXT("[CameraStreamer] Failed to bind port " << port << ": " << std::strerror(errno));
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
    server.send(theCameraImage, payload);
  }

  cameraStream.isStreaming      = true;
  cameraStream.port             = server.port();
  cameraStream.connectedClients = server.clientCount();
}
