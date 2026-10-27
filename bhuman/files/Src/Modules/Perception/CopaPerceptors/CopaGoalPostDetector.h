/**
 * @file CopaGoalPostDetector.h
 *
 * Finds the blue goal posts of the Copa NAO CCM MX goal (150 x 80 cm, blue
 * posts, white net) in a YCbCr image. Each sampled column keeps its longest
 * vertical run of blue pixels; neighbouring columns with long runs are merged
 * into boxes, which must be tall, thin and mostly filled to count as a post.
 * The crossbar and blue jerseys are rejected because their runs are short.
 *
 * Header-only and free of B-Human types so it can be tested on the host.
 */

#pragma once

#include <algorithm>
#include <vector>

namespace CopaGoalPosts
{
  struct Params
  {
    int minBlueCb = 30;          /**< Minimum Cb - 128 of a blue pixel. */
    int minCbMinusCr = 45;       /**< Minimum Cb - Cr of a blue pixel (wood and white stay below). */
    int minY = 15;               /**< Darker pixels are noise. */
    int maxY = 235;              /**< Brighter pixels are reflections. */
    int step = 2;                /**< Sampling step in visual pixels, both axes. */
    int maxGapSamples = 2;       /**< Non-blue samples tolerated inside a run. */
    float minRunFraction = 0.18f;/**< Minimum run length relative to the image height (ball patches stay below). */
    float minAspect = 2.f;       /**< Minimum height / width of a post box. */
    float minFill = 0.6f;        /**< Minimum mean run length / box height. */
    float maxSideBlue = 0.5f;    /**< Maximum blue fraction beside the box at mid height (crossbar, walls). */
  };

  struct Box
  {
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
    float fill = 0.f;
  };

  inline bool isBlue(int y, int cb, int cr, const Params& p)
  {
    return y >= p.minY && y <= p.maxY && cb - 128 >= p.minBlueCb && cb - cr >= p.minCbMinusCr;
  }

  /** Blue fraction of the samples left and right of a box at its mid height (the larger side). */
  template<typename Pixel>
  float sideBlue(const Box& box, int width, Pixel pixel, const Params& p)
  {
    const int step = std::max(1, p.step);
    const int reach = std::max(8 * step, 2 * (box.right - box.left));
    const int y = (box.top + box.bottom) / 2;
    float worst = 0.f;
    for(int side = -1; side <= 1; side += 2)
    {
      int blue = 0, total = 0;
      for(int d = step; d <= reach; d += step)
      {
        const int x = side < 0 ? box.left - d : box.right - 1 + d;
        if(x < 0 || x >= width)
          break;
        int Y, cb, cr;
        pixel(x, y, Y, cb, cr);
        blue += isBlue(Y, cb, cr, p);
        ++total;
      }
      if(total > 0)
        worst = std::max(worst, static_cast<float>(blue) / static_cast<float>(total));
    }
    return worst;
  }

  /**
   * @param pixel callable (x, y, int& Y, int& Cb, int& Cr) in visual pixels.
   * @param bottomLimit callable (x) -> first row hidden by the own body.
   * @return the post boxes, left to right.
   */
  template<typename Pixel, typename Limit>
  std::vector<Box> detect(int width, int height, Pixel pixel, Limit bottomLimit, const Params& p)
  {
    struct Run { int x, top, bottom; };
    const int step = std::max(1, p.step);
    const int minRun = std::max(4 * step, static_cast<int>(p.minRunFraction * height));
    std::vector<Run> runs;

    for(int x = 0; x < width; x += step)
    {
      const int yMax = std::min(height, bottomLimit(x));
      int start = -1, last = -1, gap = 0, bestTop = 0, bestBottom = -1;
      auto close = [&]
      {
        if(start >= 0 && last + step - start > bestBottom - bestTop)
        {
          bestTop = start;
          bestBottom = last + step;
        }
        start = -1;
      };
      for(int y = 0; y < yMax; y += step)
      {
        int Y, cb, cr;
        pixel(x, y, Y, cb, cr);
        if(isBlue(Y, cb, cr, p))
        {
          if(start < 0)
            start = y;
          last = y;
          gap = 0;
        }
        else if(start >= 0 && ++gap > p.maxGapSamples)
          close();
      }
      close();
      if(bestBottom - bestTop >= minRun)
        runs.push_back({x, bestTop, bestBottom});
    }

    std::vector<Box> boxes;
    std::size_t i = 0;
    while(i < runs.size())
    {
      Box box{runs[i].x, runs[i].x + step, runs[i].top, runs[i].bottom, 0.f};
      int lengthSum = runs[i].bottom - runs[i].top, count = 1;
      std::size_t j = i + 1;
      // One missing column is allowed; runs must overlap vertically to belong together.
      while(j < runs.size() && runs[j].x - runs[j - 1].x <= 2 * step
            && runs[j].top < box.bottom && runs[j].bottom > box.top)
      {
        box.right = runs[j].x + step;
        box.top = std::min(box.top, runs[j].top);
        box.bottom = std::max(box.bottom, runs[j].bottom);
        lengthSum += runs[j].bottom - runs[j].top;
        ++count;
        ++j;
      }
      const int h = box.bottom - box.top;
      const int w = box.right - box.left;
      box.fill = static_cast<float>(lengthSum) / static_cast<float>(count * std::max(1, h));
      if(h >= minRun && h >= p.minAspect * w && box.fill >= p.minFill
         && sideBlue(box, width, pixel, p) <= p.maxSideBlue)
        boxes.push_back(box);
      i = j;
    }
    return boxes;
  }

  /** Index pair (a, b) of the two posts whose distance is closest to the goal width, or (-1, -1). */
  template<typename Distance>
  std::pair<int, int> pairGoal(int count, Distance distance, float goalWidth, float tolerance)
  {
    std::pair<int, int> best(-1, -1);
    float bestError = tolerance * goalWidth;
    for(int a = 0; a < count; ++a)
      for(int b = a + 1; b < count; ++b)
      {
        const float d = distance(a, b);
        if(d < 0.f)
          continue;
        const float error = d > goalWidth ? d - goalWidth : goalWidth - d;
        if(error <= bestError)
        {
          bestError = error;
          best = {a, b};
        }
      }
    return best;
  }
}
