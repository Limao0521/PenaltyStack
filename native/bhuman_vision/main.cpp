// Sensor-only host. Original algorithm bodies are generated verbatim from
// heritage/bhuman_robot_perception, never patched in the immutable originals.
#include <Eigen/Core>
#include <onnxruntime_cxx_api.h>
#include <json.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <emmintrin.h>

using Vector2f = Eigen::Vector2f;
using Vector2i = Eigen::Vector2i;
using json = nlohmann::json;
#include "parameters.h"
#include "provenance.h"
#define ASSERT(expression) assert(expression)
#define STOPWATCH(name)
#define SEND_DEBUG_IMAGE(...)

namespace PixelTypes
{
  using GrayscaledPixel = unsigned char;
  struct YUYVPixel { unsigned char y0, u, y1, v; };
}

// Minimal platform declarations. Extra aligned storage is required by the
// original SIMD shrinkY, whose intermediate buffer is larger than the result.
template<typename T> class Image
{
  struct Deleter
  {
    void operator()(T* p) const
    {
#ifdef _WIN32
      _aligned_free(p);
#else
      std::free(p);
#endif
    }
  };
  std::unique_ptr<T, Deleter> storage;
public:
  unsigned width = 0, height = 0;
  void setResolution(unsigned w, unsigned h)
  {
    if(!storage)
    {
      void* p = nullptr;
#ifdef _WIN32
      p = _aligned_malloc((640 * 480 + 64) * sizeof(T), 16);
#else
      if(posix_memalign(&p, 16, (640 * 480 + 64) * sizeof(T))) p = nullptr;
#endif
      if(!p) throw std::bad_alloc();
      storage.reset(static_cast<T*>(p));
      std::memset(p, 0, (640 * 480 + 64) * sizeof(T));
    }
    if(w * h > 640 * 480) throw std::runtime_error("image too large");
    width = w; height = h;
  }
  T* operator[](unsigned y) { return storage.get() + y * width; }
  const T* operator[](unsigned y) const { return storage.get() + y * width; }
};

namespace Resize
{
  void shrinkY(unsigned downScales, const Image<unsigned char>& src, unsigned char* dest);
  void shrinkY(unsigned downScales, const Image<unsigned char>& src, Image<unsigned char>& dest)
  {
    dest.setResolution(src.width >> downScales, src.height >> downScales);
    shrinkY(downScales, src, dest[0]);
  }
}

struct LabelImage
{
  struct Annotation
  {
    Vector2f upperLeft, lowerRight;
    float probability = 0, distance = 0;
    bool isInside(const Annotation&) const;
    bool operator<(const Annotation&) const;
    float getArea() const;
    float getOverlap(const Annotation&) const;
    float getIou(const Annotation&) const;
  };
  std::vector<Annotation> annotations;
  void nonMaximumSuppression(float threshold);
  void bigBoxSuppression();
};

struct ObstaclesImagePercept
{
  struct Obstacle
  {
    int top = 0, bottom = 0, left = 0, right = 0;
    bool bottomFound = false, fallen = false;
  };
};

class LocalNetwork
{
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "bhuman_robot_pixels"};
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::array<int64_t, 4> shape{1, 60, 80, 3};
  std::array<unsigned char, 60 * 80 * 3> bytes{};
  std::array<float, 60 * 80 * 3> floats{};
  std::vector<Ort::Value> results;
  std::string inputName, outputName;
public:
  explicit LocalNetwork(const std::filesystem::path& path)
  {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, path.c_str(), options);
    if(session.GetInputCount() != 1 || session.GetOutputCount() != 1)
      throw std::runtime_error("unexpected model input/output count");
    if(session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, 60, 80, 3} ||
       session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, 8, 10, 4, 5})
      throw std::runtime_error("unexpected model shape");
    Ort::AllocatorWithDefaultOptions allocator;
    char* name = session.GetInputName(0, allocator);
    inputName = name; allocator.Free(name);
    name = session.GetOutputName(0, allocator);
    outputName = name; allocator.Free(name);
  }
  std::span<float> input(unsigned) { return {reinterpret_cast<float*>(bytes.data()), floats.size()}; }
  std::span<float> output(unsigned) { return {results.at(0).GetTensorMutableData<float>(), 8 * 10 * 4 * 5}; }
  void apply()
  {
    // Same uint8 -> float conversion as upstream CompiledNN2ONNX::apply.
    std::transform(bytes.begin(), bytes.end(), floats.begin(), [](unsigned char b) { return static_cast<float>(b); });
    auto tensor = Ort::Value::CreateTensor<float>(memory, floats.data(), floats.size(), shape.data(), shape.size());
    const char* inputs[]{inputName.c_str()};
    const char* outputs[]{outputName.c_str()};
    results = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    for(float v : output(0))
      if(!std::isfinite(v)) throw std::runtime_error("nonfinite model output");
  }
};

class RobotDetector
{
  struct { unsigned width = 640, height = 480; } theCameraInfo;
  struct { Image<unsigned char> grayscaled, saturated; } theECImage;
  Image<PixelTypes::YUYVPixel> theCameraImage;
  // No field-boundary provider is run in this isolated component. Image/ground
  // validity is checked by the calibrated consumer, and is traced explicitly.
  struct { int getBoundaryY(int) const { return 0; } } theFieldBoundary;
  struct { int getBottom(int, int height) const { return height; } } theBodyContour;
  NetworkParameters networkParameters;
  Vector2i inputImageSize{80, 60};
  LocalNetwork network;
  LocalNetwork& onnxConvModel;
  LocalNetwork& cnnConvModel;
  bool useOnnx = true;
  Image<unsigned char> yThumbnail, uThumbnail, vThumbnail, uSubsampled, vSubsampled;
  float objectThres = .6f;
  enum Classification { Horizontal, Vertical, Mixed, Nothing, Unknown, Default };
  struct Region
  {
    Vector2i regionIndices;
    std::vector<int> contrastChanges = {0, 0};
    Classification classification = Nothing;
    bool clustered = false, bright = false;
    int brightSpots = 0, maxY = -1;
  };
  unsigned xyStep = 2, xyRegions = 16, minNeighborPoints = 5;
  short minContrastDiff = 40, brightnessThreshold = 100;
  unsigned char satThreshold = 40;
  float mixedThresh = .2f;
  int minHetSpots = 2;
  // Feet are projected and width/visibility checked in the calibrated consumer.
  bool trimObstacles = false;
  bool trimObstacle(bool, ObstaclesImagePercept::Obstacle&) { throw std::runtime_error("trim provider unavailable"); }
  void scanImage(std::vector<std::vector<Region>>&);
  void classifyRegions(std::vector<std::vector<Region>>&);
  void discardHomogeneousAreas(std::vector<std::vector<Region>>&);
  void dbScan(std::vector<std::vector<Region>>&, std::vector<ObstaclesImagePercept::Obstacle>&);
  bool expandCluster(std::vector<std::vector<Region>>&, Region&, std::vector<Vector2i>&,
                     std::vector<Vector2i>&, Vector2i&, Vector2i&);
  void regionQuery(std::vector<std::vector<Region>>&, Vector2i&, int,
                   std::vector<Vector2i>&, Classification classificationToSearch = Default);
  void fillGrayscaleThumbnail();
  void fillUVThumbnails();
  void applyColorNetwork();
  template<typename ConvModel> void boundingBoxes(LabelImage&, ConvModel&);
  LabelImage::Annotation predictionToBoundingBox(Eigen::Map<Eigen::Vector<float, 5>>& pred,
                                                unsigned y, unsigned x, unsigned b) const;
public:
  float maxConfidence = 0.f;
  explicit RobotDetector(const std::filesystem::path& path) : network(path), onnxConvModel(network), cnnConvModel(network) {}
  json detect(const std::vector<unsigned char>& bytes, const std::string& format, unsigned width, unsigned height,
              const std::string& camera);
};

#include "original_algorithms.h"

json RobotDetector::detect(const std::vector<unsigned char>& bytes, const std::string& format,
                           unsigned width, unsigned height, const std::string& camera)
{
  if(!((width == 640 && height == 480) || (width == 320 && height == 240)))
    throw std::runtime_error("upper image must be VGA or QVGA, 4:3, unmirrored");
  if((format != "rgb" && format != "yuyv") || bytes.size() != width * height * (format == "rgb" ? 3 : 2))
    throw std::runtime_error("invalid pixel format or byte count");
  theCameraInfo.width = width; theCameraInfo.height = height;
  theECImage.grayscaled.setResolution(width, height);
  theECImage.saturated.setResolution(width, height);
  theCameraImage.setResolution(width / 2, height);
  for(unsigned y = 0; y < height; ++y)
    for(unsigned x = 0; x < width; x += 2)
    {
      PixelTypes::YUYVPixel& p = theCameraImage[y][x / 2];
      if(format == "yuyv")
        std::memcpy(&p, bytes.data() + (y * width + x) * 2, 4);
      else
      {
        const unsigned char* rgb = bytes.data() + (y * width + x) * 3;
        unsigned char u0, v0, u1, v1;
        ColorModelConversions::fromRGBToYUV(rgb[0], rgb[1], rgb[2], p.y0, u0, v0);
        ColorModelConversions::fromRGBToYUV(rgb[3], rgb[4], rgb[5], p.y1, u1, v1);
        p.u = static_cast<unsigned char>((static_cast<unsigned>(u0) + u1) / 2);
        p.v = static_cast<unsigned char>((static_cast<unsigned>(v0) + v1) / 2);
      }
      theECImage.grayscaled[y][x] = p.y0;
      theECImage.grayscaled[y][x + 1] = p.y1;
      const int u = p.u - 128, v = p.v - 128;
      // Scalar lighting-independent saturation input adapter (ECImageProvider).
      // Original JIT uses reciprocal approximations; this is not bit-identical.
      const float norm = 256.f * std::sqrt(2.f * (u*u + v*v));
      theECImage.saturated[y][x] = static_cast<unsigned char>(std::min(255.f, std::round(norm / std::max(1, static_cast<int>(p.y0)))));
      theECImage.saturated[y][x+1] = static_cast<unsigned char>(std::min(255.f, std::round(norm / std::max(1, static_cast<int>(p.y1)))));
    }
  if(camera == "lower")
  {
    std::vector<std::vector<Region>> regions(xyRegions, std::vector<Region>(xyRegions, Region()));
    std::vector<ObstaclesImagePercept::Obstacle> obstacles;
    scanImage(regions);
    classifyRegions(regions);
    discardHomogeneousAreas(regions);
    dbScan(regions, obstacles);
    json detections = json::array();
    for(const auto& o : obstacles)
      detections.push_back({{"box", {o.left, o.top, o.right, o.bottom}}, {"confidence", 1.0},
                            {"confidence_measured", false}});
    maxConfidence = 0.f;
    return detections;
  }
  if(camera != "upper") throw std::runtime_error("camera must be upper or lower");
  applyColorNetwork();
  maxConfidence = 0.f;
  const auto rawOutput = onnxConvModel.output(0);
  for(size_t i = 0; i < rawOutput.size(); i += 5)
    maxConfidence = std::max(maxConfidence, 1.f / (1.f + std::exp(-rawOutput[i])));
  LabelImage boxes;
  boundingBoxes(boxes, onnxConvModel);
  boxes.nonMaximumSuppression(.3f);
  boxes.bigBoxSuppression();
  json detections = json::array();
  for(const auto& box : boxes.annotations)
    detections.push_back({{"box", {box.upperLeft.x(), box.upperLeft.y(), box.lowerRight.x(), box.lowerRight.y()}},
                          {"confidence", box.probability}});
  return detections;
}

static std::vector<unsigned char> decodeBase64(const std::string& encoded)
{
  static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if(encoded.size() % 4 || encoded.size() > 640 * 480 * 4)
    throw std::runtime_error("invalid base64 length");
  std::vector<unsigned char> result;
  result.reserve(encoded.size() / 4 * 3);
  for(size_t i = 0; i < encoded.size(); i += 4)
  {
    unsigned value = 0, padding = 0;
    for(unsigned j = 0; j < 4; ++j)
    {
      const char c = encoded[i + j];
      if(c == '=')
      {
        if(i + 4 != encoded.size() || j < 2) throw std::runtime_error("invalid base64 padding");
        ++padding; value <<= 6;
      }
      else
      {
        const size_t digit = alphabet.find(c);
        if(digit == std::string::npos || padding) throw std::runtime_error("invalid base64 digit");
        value = (value << 6) | static_cast<unsigned>(digit);
      }
    }
    result.push_back(static_cast<unsigned char>(value >> 16));
    if(padding < 2) result.push_back(static_cast<unsigned char>(value >> 8));
    if(!padding) result.push_back(static_cast<unsigned char>(value));
  }
  return result;
}

int main(int argc, char** argv)
{
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  try
  {
    if(argc != 2) throw std::runtime_error("usage: bhuman_vision <original_model.onnx>");
    RobotDetector detector{std::filesystem::path(argv[1])};
    json hello = json::parse(provenance_json);
    hello["type"] = "HELLO";
    std::cout << hello.dump() << std::endl;
    long long expectedSequence = 0;
    for(;;)
    {
      // Bounded before allocation/parsing, even for a corrupt producer.
      std::string line;
      while(true)
      {
        const int c = std::cin.get();
        if(c == EOF) return 0;
        if(c == '\n') break;
        if(line.size() >= 1300000) throw std::runtime_error("input frame too large");
        line += static_cast<char>(c);
      }
      const json request = json::parse(line);
      if(request.value("type", "") == "SHUTDOWN") return 0;
      if(request.at("version") != "bhuman_robot_pixels_v1" || request.at("type") != "IMAGE" ||
         !request.at("sequence").is_number_integer() || request.at("sequence").get<long long>() != expectedSequence ||
         !request.at("capture_s").is_number() || !std::isfinite(request.at("capture_s").get<double>()))
        throw std::runtime_error("invalid protocol, sequence or capture time");
      const auto begin = std::chrono::steady_clock::now();
      const auto bytes = decodeBase64(request.at("pixels_b64").get<std::string>());
      const std::string camera = request.value("camera", "upper");
      json boxes = detector.detect(bytes, request.at("format"), request.at("width"), request.at("height"), camera);
      const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      std::cout << json{{"type", "DETECTIONS"}, {"version", "bhuman_robot_pixels_v1"},
                        {"sequence", expectedSequence++}, {"capture_s", request.at("capture_s")},
                        {"width", request.at("width")}, {"height", request.at("height")},
                        {"camera", camera}, {"method", camera == "upper" ? "original_upper_network" : "original_lower_contrast"},
                        {"boxes", boxes}, {"inference_ms", elapsed},
                        {"max_confidence", camera == "upper" ? json(detector.maxConfidence) : json(nullptr)},
                        {"threshold", camera == "upper" ? json(.6) : json(nullptr)}, {"actuators", false}}.dump() << std::endl;
    }
  }
  catch(const std::exception& error)
  {
    std::cout << json{{"type", "ERROR"}, {"message", error.what()}}.dump() << std::endl;
    return 1;
  }
}
