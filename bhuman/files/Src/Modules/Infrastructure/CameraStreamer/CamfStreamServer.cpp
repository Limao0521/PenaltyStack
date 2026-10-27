/**
 * @file CamfStreamServer.cpp
 *
 * The CameraImage is stored as YUYV (4:2:2): each pair of visual pixels
 * shares one U and one V sample.  JPEGImage encodes this raw data as a
 * fake 4-channel JCS_CMYK JPEG (no color conversion), which viewers like
 * OpenCV/cv2.imdecode cannot display correctly (colors appear as purple/green).
 * This server encodes the image itself using libjpeg with JCS_YCbCr input,
 * producing a standard JPEG that any viewer can display.
 */

#include "CamfStreamServer.h"
#include "Streaming/Output.h"
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <jpeglib.h>

static const unsigned char MAGIC[4] = {'C', 'A', 'M', 'F'};

// Called when the output buffer is full — should never happen with our
// pre-allocated buffer sized at raw image bytes (always > any JPEG output).
static boolean csOnDestEmpty(j_compress_ptr) { return false; }
static void csOnDestIgnore(j_compress_ptr) {}

CamfStreamServer::~CamfStreamServer()
{
  for(int fd : clients)
    ::close(fd);
  if(serverFd >= 0)
    ::close(serverFd);
}

bool CamfStreamServer::start(int port)
{
  serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if(serverFd < 0)
    return false;

  // Allow fast restart after crash
  int optval = 1;
  ::setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  // Non-blocking accept
  ::fcntl(serverFd, F_SETFL, O_NONBLOCK);

  sockaddr_in addr{};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port        = htons(static_cast<uint16_t>(port));

  if(::bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
  {
    ::close(serverFd);
    serverFd = -1;
    return false;
  }

  ::listen(serverFd, 4);
  activePort = port;
  return true;
}

void CamfStreamServer::acceptClients()
{
  for(;;)
  {
    const int clientFd = ::accept(serverFd, nullptr, nullptr);
    if(clientFd < 0)
      break; // EAGAIN / EWOULDBLOCK — no more pending connections

    // Disable Nagle: send small headers + large JPEG payload without delay
    int flag = 1;
    ::setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Set send timeout so a slow/dead client doesn't block the camera thread.
    // Must stay well below MotionEngine's emergencySitDownDelay (3 s): the
    // sends run in the camera thread, and a stalled client would otherwise
    // freeze frames long enough to trigger the "I'm blind" sit-down.
    struct timeval tv{ .tv_sec = 0, .tv_usec = 100000 };
    ::setsockopt(clientFd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    clients.push_back(clientFd);
    OUTPUT_TEXT("[CameraStreamer] New client on port " << activePort
                << " (total=" << static_cast<unsigned int>(clients.size()) << ")");
  }
}

unsigned CamfStreamServer::encode(const CameraImage& image)
{
  // Visual image dimensions
  const int vW = static_cast<int>(image.width) * 2;  // pairs × 2
  const int vH = static_cast<int>(image.height);
  if(vW <= 0 || vH <= 0)
    return 0;

  // Pre-allocate buffers (lazy, only grows)
  const size_t maxJpeg = static_cast<size_t>(vW * vH * 3); // upper bound
  if(jpegBuf.size() < maxJpeg)
    jpegBuf.resize(maxJpeg);

  const size_t rowBytes = static_cast<size_t>(vW * 3); // Y,Cb,Cr per pixel
  if(rowBuf.size() < rowBytes)
    rowBuf.resize(rowBytes);

  jpeg_compress_struct cInfo;
  jpeg_error_mgr jem;
  cInfo.err = jpeg_std_error(&jem);
  jpeg_create_compress(&cInfo);

  // Set up in-memory destination (same pattern as JPEGImage.cpp)
  if(!cInfo.dest)
    cInfo.dest = static_cast<jpeg_destination_mgr*>(
                 (*cInfo.mem->alloc_small)(reinterpret_cast<j_common_ptr>(&cInfo),
                                           JPOOL_PERMANENT,
                                           sizeof(jpeg_destination_mgr)));
  cInfo.dest->init_destination    = csOnDestIgnore;
  cInfo.dest->empty_output_buffer = csOnDestEmpty;
  cInfo.dest->term_destination    = csOnDestIgnore;
  cInfo.dest->next_output_byte    = static_cast<JOCTET*>(jpegBuf.data());
  cInfo.dest->free_in_buffer      = jpegBuf.size();

  cInfo.image_width      = static_cast<JDIMENSION>(vW);
  cInfo.image_height     = static_cast<JDIMENSION>(vH);
  cInfo.input_components = 3;
  cInfo.in_color_space   = JCS_YCbCr; // standard displayable color space

  jpeg_set_defaults(&cInfo);
  cInfo.dct_method = JDCT_FASTEST;
  jpeg_set_quality(&cInfo, 75, true);
  jpeg_start_compress(&cInfo, true);

  while(cInfo.next_scanline < cInfo.image_height)
  {
    const int y = static_cast<int>(cInfo.next_scanline);
    unsigned char* dst = rowBuf.data();

    // Each image column x holds a YUYV pair (y0, u, y1, v)
    // representing visual pixels (2x, y) and (2x+1, y).
    // libjpeg JCS_YCbCr expects [Y, Cb, Cr] per visual pixel.
    for(int x = 0; x < static_cast<int>(image.width); ++x)
    {
      const auto& px = image(static_cast<unsigned int>(x), static_cast<unsigned int>(y));

      // Visual pixel 2x
      *dst++ = px.y0;
      *dst++ = px.u;   // Cb
      *dst++ = px.v;   // Cr

      // Visual pixel 2x+1 (shares U/V with 2x — 4:2:2 → 4:4:4 upsampling)
      *dst++ = px.y1;
      *dst++ = px.u;
      *dst++ = px.v;
    }

    JSAMPROW rowPtr = rowBuf.data();
    jpeg_write_scanlines(&cInfo, &rowPtr, 1);
  }

  jpeg_finish_compress(&cInfo);
  const unsigned sz = static_cast<unsigned>(
      reinterpret_cast<unsigned char*>(cInfo.dest->next_output_byte) - jpegBuf.data());
  jpeg_destroy_compress(&cInfo);
  return sz;
}

void CamfStreamServer::send(const CameraImage& image, const std::vector<unsigned char>& payload)
{
  const unsigned sz = encode(image);
  if(sz == 0)
    return;

  unsigned char header[8];
  std::memcpy(header, MAGIC, 4);
  header[4] = static_cast<unsigned char>( sz        & 0xFF);
  header[5] = static_cast<unsigned char>((sz >>  8) & 0xFF);
  header[6] = static_cast<unsigned char>((sz >> 16) & 0xFF);
  header[7] = static_cast<unsigned char>((sz >> 24) & 0xFF);

  std::vector<int> toRemove;
  for(int fd : clients)
  {
    if(!sendAll(fd, header, 8) ||
       !sendAll(fd, jpegBuf.data(), static_cast<int>(sz)) ||
       !sendAll(fd, payload.data(), static_cast<int>(payload.size())))
    {
      ::close(fd);
      toRemove.push_back(fd);
    }
  }
  for(int fd : toRemove)
  {
    clients.erase(std::remove(clients.begin(), clients.end(), fd), clients.end());
    OUTPUT_TEXT("[CameraStreamer] Client disconnected (remaining=" << static_cast<unsigned int>(clients.size()) << ")");
  }
}

void CamfStreamServer::appendBallPayload(std::vector<unsigned char>& out, const BallPercept& ballPercept,
                                         const BallSpots& ballSpots, const RawBallPatch& rawBallPatch)
{
  // status: 0=notSeen  1=seen  2=guessed  (matches BallPercept::Status enum)
  append(out, static_cast<unsigned char>(ballPercept.status));
  append(out, ballPercept.positionInImage.x());
  append(out, ballPercept.positionInImage.y());
  append(out, ballPercept.radiusInImage);

  // BallSpots are candidates BEFORE classification — useful to diagnose whether
  // the pipeline even proposes the trionda as a candidate region.
  const auto& spots = ballSpots.ballSpots;
  const uint8_t numSpots = static_cast<uint8_t>(std::min(spots.size(), size_t(50)));
  append(out, numSpots);
  for(std::size_t i = 0; i < numSpots; ++i)
  {
    append(out, static_cast<int32_t>(spots[i].x()));
    append(out, static_cast<int32_t>(spots[i].y()));
  }

  const bool patchValid = rawBallPatch.valid && !rawBallPatch.data.empty();
  append(out, static_cast<unsigned char>(patchValid ? 1 : 0));
  if(patchValid)
  {
    append(out, static_cast<uint16_t>(rawBallPatch.patchSize));
    const std::size_t size = out.size();
    out.resize(size + rawBallPatch.data.size() * 4u);
    std::memcpy(out.data() + size, rawBallPatch.data.data(), rawBallPatch.data.size() * 4u);
  }
}

bool CamfStreamServer::sendAll(int fd, const unsigned char* data, int len)
{
  int sent = 0;
  while(sent < len)
  {
    const int n = static_cast<int>(::send(fd, data + sent, static_cast<size_t>(len - sent), 0));
    if(n > 0)
      sent += n;
    else
      return false; // disconnected or send timeout
  }
  return true;
}
