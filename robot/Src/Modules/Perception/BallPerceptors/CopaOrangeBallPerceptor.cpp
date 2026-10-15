/**
 * @file CopaOrangeBallPerceptor.cpp
 *
 * @author PenaltyStack
 */

#include "CopaOrangeBallPerceptor.h"
#include "Tools/Math/Projection.h"
#include "Math/Geometry.h"
#include <algorithm>
#include <cmath>

MAKE_MODULE(CopaOrangeBallPerceptor);

bool CopaOrangeBallPerceptor::isOrange(int x, int y) const
{
  const int luminance = theCameraImage.getY(x, y);
  const int u = theCameraImage.getU(x, y);
  const int v = theCameraImage.getV(x, y);
  return luminance >= minY && luminance <= maxY && u <= maxU && v >= minV && v - u >= minChroma;
}

void CopaOrangeBallPerceptor::update(BallPercept& ballPercept)
{
  ballPercept.status = BallPercept::notSeen;
  const bool upper = theCameraInfo.camera == CameraInfo::upper;
  if(!theCameraMatrix.isValid || (upper && !useUpperCamera) || (!upper && !useLowerCamera))
    return;

  const int step = std::max(1, gridStep);
  const int width = theCameraInfo.width, height = theCameraInfo.height;
  const int gridWidth = width / step, gridHeight = height / step;
  if(gridWidth < 2 || gridHeight < 2)
    return;
  mask.assign(static_cast<size_t>(gridWidth * gridHeight), 0);
  for(int gx = 0; gx < gridWidth; ++gx)
  {
    const int x = gx * step + step / 2;
    int yMax = height;
    theBodyContour.clipBottom(x, yMax); // the robot's own feet and legs are not the ball
    for(int gy = 0; gy < gridHeight; ++gy)
    {
      const int y = gy * step + step / 2;
      if(y >= yMax)
        break;
      if(isOrange(x, y))
        mask[gy * gridWidth + gx] = 1;
    }
  }

  float bestScore = 0.f;
  for(int start = 0; start < gridWidth * gridHeight; ++start)
  {
    if(mask[start] != 1)
      continue;
    // Flood fill (4-neighbourhood) of one orange blob.
    queue.clear();
    queue.push_back(start);
    mask[start] = 2;
    int minX = gridWidth, maxX = -1, minYCell = gridHeight, maxYCell = -1;
    for(size_t i = 0; i < queue.size(); ++i)
    {
      const int cell = queue[i];
      const int cx = cell % gridWidth, cy = cell / gridWidth;
      minX = std::min(minX, cx);
      maxX = std::max(maxX, cx);
      minYCell = std::min(minYCell, cy);
      maxYCell = std::max(maxYCell, cy);
      const int neighbours[4][2] = {{cx - 1, cy}, {cx + 1, cy}, {cx, cy - 1}, {cx, cy + 1}};
      for(const auto& n : neighbours)
        if(n[0] >= 0 && n[0] < gridWidth && n[1] >= 0 && n[1] < gridHeight && mask[n[1] * gridWidth + n[0]] == 1)
        {
          mask[n[1] * gridWidth + n[0]] = 2;
          queue.push_back(n[1] * gridWidth + n[0]);
        }
    }
    const int cells = static_cast<int>(queue.size());
    const int boxWidth = maxX - minX + 1, boxHeight = maxYCell - minYCell + 1;
    const float fill = static_cast<float>(cells) / static_cast<float>(boxWidth * boxHeight);
    const float aspect = static_cast<float>(std::max(boxWidth, boxHeight)) / static_cast<float>(std::min(boxWidth, boxHeight));
    if(cells < minBlobCells || fill < minFillRatio || aspect > maxAspectRatio)
      continue;

    const Vector2f center(static_cast<float>((minX + maxX + 1) * step) / 2.f, static_cast<float>((minYCell + maxYCell + 1) * step) / 2.f);
    const float radius = static_cast<float>(std::max(boxWidth, boxHeight) * step) / 2.f;
    Vector2f onField;
    Matrix2f covariance;
    if(!theMeasurementCovariance.transformWithCovLegacy(center, theBallSpecification.radius, rotationDeviation, onField, covariance) ||
       onField.norm() > maxDistance)
      continue;
    // Orange objects that are not a 65 mm ball on the floor (e.g. shell parts of the other robot)
    // project to a distance at which their image size does not match: reject or down-weight them.
    float sizeMismatch = 0.f;
    Geometry::Circle expected;
    if(Projection::calculateBallInImage(onField, theCameraMatrix, theCameraInfo, theBallSpecification.radius, expected) &&
       expected.radius > 0.f)
    {
      const float ratio = radius / expected.radius;
      if(ratio < minRadiusRatio || ratio > maxRadiusRatio)
        continue;
      sizeMismatch = std::abs(std::log(ratio));
    }
    const float score = static_cast<float>(cells) * fill / (1.f + 2.f * sizeMismatch);
    if(score <= bestScore)
      continue;
    bestScore = score;
    ballPercept.positionInImage = center;
    ballPercept.radiusInImage = radius;
    ballPercept.positionOnField = onField;
    ballPercept.radiusOnField = theBallSpecification.radius;
    ballPercept.covarianceOnField = covariance;
    ballPercept.status = BallPercept::seen;
  }
}
