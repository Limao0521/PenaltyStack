/**
 * @file CopaVisionPerceptor.cpp
 */

#include "CopaVisionPerceptor.h"
#include "CopaGoalPostDetector.h"
#include "CopaVisionShared.h"
#include "Streaming/Output.h"
#include "Tools/Math/Transformation.h"
#include <cmath>
#include <sstream>

MAKE_MODULE(CopaVisionPerceptor);

void CopaVisionPerceptor::update(CopaVisionPercept& percept)
{
  percept.upper = theCameraInfo.camera == CameraInfo::upper;
  percept.posts.clear();
  percept.robots.clear();
  percept.goalSeen = false;
  percept.goalLeftPost = percept.goalRightPost = -1;
  percept.goalSpan = percept.goalDistance = percept.goalCenterBearing = 0.f;

  const int width = static_cast<int>(theCameraImage.width) * 2;
  const int height = static_cast<int>(theCameraImage.height);
  if(width <= 0 || height <= 0)
    return;

  CopaGoalPosts::Params params;
  params.minBlueCb = minBlueCb;
  params.minCbMinusCr = minCbMinusCr;
  params.minRunFraction = minRunFraction;
  params.minAspect = minAspect;
  auto pixel = [this](int x, int y, int& Y, int& cb, int& cr)
  {
    const PixelTypes::YUYVPixel& p = theCameraImage(static_cast<unsigned>(x / 2), static_cast<unsigned>(y));
    Y = x & 1 ? p.y1 : p.y0;
    cb = p.u;
    cr = p.v;
  };
  auto bottomLimit = [this, height](int x) { return theBodyContour.getBottom(x, height); };

  for(const CopaGoalPosts::Box& found : CopaGoalPosts::detect(width, height, pixel, bottomLimit, params))
  {
    CopaVisionPercept::Box& post = percept.posts.emplace_back();
    post.left = found.left;
    post.right = found.right;
    post.top = found.top;
    post.bottom = found.bottom;
    post.bottomVisible = found.bottom < bottomLimit((found.left + found.right) / 2) - params.step;
    locate(post);
  }

  // Posts are ordered left to right in the image, i.e. by decreasing bearing.
  if(percept.posts.size() >= 2)
  {
    const float span = percept.posts.front().bearing - percept.posts.back().bearing;
    if(span >= minGoalSpan && span <= maxGoalSpan)
    {
      percept.goalSeen = true;
      percept.goalLeftPost = 0;
      percept.goalRightPost = static_cast<int>(percept.posts.size()) - 1;
      percept.goalSpan = span;
      percept.goalCenterBearing = 0.5f * (percept.posts.front().bearing + percept.posts.back().bearing);
      percept.goalDistance = 0.5f * goalWidth / std::tan(0.5f * span) * std::cos(percept.goalCenterBearing);
    }
  }

  for(const ObstaclesImagePercept::Obstacle& obstacle : theObstaclesImagePercept.obstacles)
  {
    if(isBall(obstacle))
      continue;
    CopaVisionPercept::Robot& robot = percept.robots.emplace_back();
    robot.box.left = obstacle.left;
    robot.box.right = obstacle.right;
    robot.box.top = obstacle.top;
    robot.box.bottom = obstacle.bottom;
    robot.box.bottomVisible = obstacle.bottomFound;
    robot.fallen = obstacle.fallen;
    locate(robot.box);
  }

  if(percept.upper)
  {
    CopaVisionSnapshot snapshot;
    snapshot.time = theFrameInfo.time;
    for(const auto& post : percept.posts)
      snapshot.postBearings.push_back(post.bearing);
    for(const auto& robot : percept.robots)
      snapshot.robotBearings.push_back(robot.box.bearing);
    CopaVisionShared::instance().publish(snapshot);
  }

  log(percept);
}

float CopaVisionPerceptor::bearing(const Vector2f& pointInImage) const
{
  const Vector2f corrected = theImageCoordinateSystem.toCorrected(pointInImage);
  const Vector3f ray(theCameraInfo.focalLength,
                     theCameraInfo.opticalCenter.x() - corrected.x(),
                     theCameraInfo.opticalCenter.y() - corrected.y());
  const Vector3f inRobot = theCameraMatrix.rotation * ray;
  return std::atan2(inRobot.y(), inRobot.x());
}

void CopaVisionPerceptor::locate(CopaVisionPercept::Box& box) const
{
  box.bearing = bearing(Vector2f(0.5f * static_cast<float>(box.left + box.right),
                                 0.5f * static_cast<float>(box.top + box.bottom)));
  box.onField = false;
  if(!box.bottomVisible)
    return;
  const Vector2f base(0.5f * static_cast<float>(box.left + box.right), static_cast<float>(box.bottom));
  Vector2f onField;
  if(Transformation::imageToRobot(theImageCoordinateSystem.toCorrected(base), theCameraMatrix, theCameraInfo, onField)
     && std::isfinite(onField.x()) && std::isfinite(onField.y()) && onField.x() > 0.f && onField.norm() <= maxDistance)
  {
    box.onField = true;
    box.positionOnField = onField;
  }
}

bool CopaVisionPerceptor::isBall(const ObstaclesImagePercept::Obstacle& obstacle) const
{
  if(theBallPercept.status != BallPercept::seen || theBallPercept.radiusInImage <= 0.f)
    return false;
  const Vector2f& ball = theBallPercept.positionInImage;
  const bool around = ball.x() >= obstacle.left && ball.x() <= obstacle.right
                      && ball.y() >= obstacle.top && ball.y() <= obstacle.bottom;
  const float limit = robotBallOverlap * 2.f * theBallPercept.radiusInImage;
  return around && obstacle.right - obstacle.left <= limit && obstacle.bottom - obstacle.top <= limit;
}

void CopaVisionPerceptor::log(const CopaVisionPercept& percept)
{
  const auto now = std::chrono::steady_clock::now();
  if(logIntervalMs <= 0 || now - lastLog < std::chrono::milliseconds(logIntervalMs))
    return;
  lastLog = now;

  auto degrees = [](float rad) { return static_cast<int>(std::lround(rad * 180.f / 3.14159265f)); };
  std::ostringstream text;
  text << "[CopaVision] camera=" << (percept.upper ? "upper" : "lower") << " posts=" << percept.posts.size();
  for(const auto& post : percept.posts)
    text << " post@" << degrees(post.bearing) << "deg";
  text << " goal=" << (percept.goalSeen ? "yes" : "no");
  if(percept.goalSeen)
    text << " span=" << degrees(percept.goalSpan) << "deg distance=" << static_cast<int>(percept.goalDistance)
         << "mm center=" << degrees(percept.goalCenterBearing) << "deg";
  text << " robots=" << percept.robots.size();
  for(const auto& robot : percept.robots)
    text << " robot@" << degrees(robot.box.bearing) << "deg" << (robot.fallen ? "(fallen)" : "");
  OUTPUT_WARNING(text.str());
}
