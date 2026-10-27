/**
 * @file CopaVisionShared.h
 *
 * Latest upper-camera post and robot bearings, handed from CopaVisionPerceptor
 * (Upper thread) to the Copa striker behavior (Cognition thread).
 *
 * A shared snapshot instead of a Cognition representation: SkillBehaviorControl
 * runs in every scenario, and requiring a Copa-only representation there would
 * stop B-Human in all non-Copa scenarios. Same pattern as RLSharedState.
 */

#pragma once

#include <mutex>
#include <vector>

struct CopaVisionSnapshot
{
  unsigned time = 0;               /**< FrameInfo time of the image (ms). */
  std::vector<float> postBearings; /**< Robot-relative, radians, left positive. */
  std::vector<float> robotBearings;
};

class CopaVisionShared
{
public:
  static CopaVisionShared& instance()
  {
    static CopaVisionShared shared;
    return shared;
  }

  void publish(const CopaVisionSnapshot& snapshot)
  {
    std::lock_guard<std::mutex> lock(mutex);
    latest = snapshot;
  }

  CopaVisionSnapshot read() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    return latest;
  }

private:
  mutable std::mutex mutex;
  CopaVisionSnapshot latest;
};
