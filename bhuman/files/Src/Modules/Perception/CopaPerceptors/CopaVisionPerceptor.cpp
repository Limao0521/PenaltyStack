/**
 * @file CopaVisionPerceptor.cpp
 */

#include "CopaVisionPerceptor.h"
#include "CopaGoalPostDetector.h"
#include "Streaming/Output.h"
#include "Tools/Math/Transformation.h"
#include <cmath>
#include <sstream>

MAKE_MODULE(CopaVisionPerceptor);

void CopaVisionPerceptor::update(CopaVisionPercept& percept)
{
  percept.posts.clear();
  percept.robots.clear();
  percept.goalSeen = false;
  percept.goalLeftPost = percept.goalRightPost = -1;
  percept.goalWidth = 0.f;

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
    project(post);
  }

  auto distance = [&percept](int a, int b)
  {
    const auto& pa = percept.posts[a];
    const auto& pb = percept.posts[b];
    return pa.onField && pb.onField ? (pa.positionOnField - pb.positionOnField).norm() : -1.f;
  };
  const auto [a, b] = CopaGoalPosts::pairGoal(static_cast<int>(percept.posts.size()), distance, goalWidth, goalWidthTolerance);
  if(a >= 0)
  {
    const bool aIsLeft = percept.posts[a].positionOnField.y() > percept.posts[b].positionOnField.y();
    percept.goalSeen = true;
    percept.goalLeftPost = aIsLeft ? a : b;
    percept.goalRightPost = aIsLeft ? b : a;
    percept.goalWidth = distance(a, b);
    percept.goalCenterOnField = (percept.posts[a].positionOnField + percept.posts[b].positionOnField) * 0.5f;
  }

  for(const ObstaclesImagePercept::Obstacle& obstacle : theObstaclesImagePercept.obstacles)
  {
    CopaVisionPercept::Robot& robot = percept.robots.emplace_back();
    robot.box.left = obstacle.left;
    robot.box.right = obstacle.right;
    robot.box.top = obstacle.top;
    robot.box.bottom = obstacle.bottom;
    robot.box.bottomVisible = obstacle.bottomFound;
    robot.fallen = obstacle.fallen;
    project(robot.box);
  }

  log(percept);
}

void CopaVisionPerceptor::project(CopaVisionPercept::Box& box) const
{
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

void CopaVisionPerceptor::log(const CopaVisionPercept& percept)
{
  const auto now = std::chrono::steady_clock::now();
  if(logIntervalMs <= 0 || now - lastLog < std::chrono::milliseconds(logIntervalMs))
    return;
  lastLog = now;

  std::ostringstream text;
  text << "[CopaVision] camera=" << (theCameraInfo.camera == CameraInfo::lower ? "lower" : "upper")
       << " posts=" << percept.posts.size();
  for(const auto& post : percept.posts)
    if(post.onField)
      text << " post@" << static_cast<int>(post.positionOnField.norm()) << "mm";
  text << " goal=" << (percept.goalSeen ? "yes" : "no");
  if(percept.goalSeen)
    text << " width=" << static_cast<int>(percept.goalWidth) << "mm center=("
         << static_cast<int>(percept.goalCenterOnField.x()) << "," << static_cast<int>(percept.goalCenterOnField.y()) << ")";
  text << " robots=" << percept.robots.size();
  for(const auto& robot : percept.robots)
    if(robot.box.onField)
      text << " robot@" << static_cast<int>(robot.box.positionOnField.norm()) << "mm" << (robot.fallen ? "(fallen)" : "");
  OUTPUT_WARNING(text.str());
}
