/**
 * @file CamfStreamServer.h
 *
 * Non-blocking TCP server that sends camera frames in the CAMF protocol:
 *   [4 bytes] magic "CAMF"
 *   [4 bytes] uint32 little-endian JPEG size
 *   [N bytes] raw JPEG data (standard YCbCr JPEG)
 *   [payload] bytes chosen by the module (ball metadata, ...)
 *
 * Shared by CameraStreamer and CopaCameraStreamer.
 */

#pragma once

#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Perception/BallPercepts/BallPercept.h"
#include "Representations/Perception/BallPercepts/BallSpots.h"
#include "Representations/Perception/BallPercepts/RawBallPatch.h"
#include <cstring>
#include <vector>

class CamfStreamServer
{
public:
  ~CamfStreamServer();

  /** Bind and listen on the given port. Returns true on success. */
  bool start(int port);
  bool listening() const { return serverFd >= 0; }
  int port() const { return activePort; }
  int clientCount() const { return static_cast<int>(clients.size()); }

  /** Accept any pending connections (non-blocking). */
  void acceptClients();

  /** Encode the image as JPEG and send header + JPEG + payload to all clients. */
  void send(const CameraImage& image, const std::vector<unsigned char>& payload);

  /**
   * Ball metadata following the JPEG:
   *   uint8 status | float32 x | float32 y | float32 radius             (BallPercept)
   *   uint8 count | (int32 x, int32 y) x count                         (BallSpots, max 50)
   *   uint8 valid | [uint16 patchSize | float32 x patchSize^2*3]        (RawBallPatch)
   */
  static void appendBallPayload(std::vector<unsigned char>& out, const BallPercept& ballPercept,
                                const BallSpots& ballSpots, const RawBallPatch& rawBallPatch);

  template<typename T>
  static void append(std::vector<unsigned char>& out, const T& value)
  {
    const std::size_t size = out.size();
    out.resize(size + sizeof(T));
    std::memcpy(out.data() + size, &value, sizeof(T));
  }

private:
  int serverFd = -1;                  /**< Listening server socket. */
  int activePort = 0;                 /**< Port this instance is bound to. */
  std::vector<int> clients;           /**< Connected client sockets. */
  std::vector<unsigned char> jpegBuf; /**< Pre-allocated JPEG output buffer. */
  std::vector<unsigned char> rowBuf;  /**< Per-scanline YCbCr row buffer. */

  /** Returns the JPEG size, 0 on failure. */
  unsigned encode(const CameraImage& image);

  /** Send exactly `len` bytes to `fd`; returns false if the client disconnected. */
  static bool sendAll(int fd, const unsigned char* data, int len);
};
