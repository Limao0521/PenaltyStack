/**
 * @file CopaOrangeBall.h
 *
 * Color detector for the official Copa NAO CCM MX 2026 ball (§ 3.3.4: orange
 * plastic, 65 mm, 55 g) on the wood floor (duela). The YOLO models were trained
 * on the Trionda and do not know a plain orange ball.
 *
 * Wood is a desaturated orange, so the ball is separated by how much more
 * orange it is than the floor of the same image: a pixel counts when its
 * Cr - Cb is above an absolute minimum AND above the floor median plus a
 * margin, with an orange hue (red and yellow rejected). Orange samples are
 * grouped in connected blobs; a blob is the ball when it is round, mostly
 * filled and its size matches the size a 65 mm ball has at that image point
 * (given by the caller from the camera matrix). Hands, jerseys and the floor
 * fail one of these.
 *
 * Header-only and free of B-Human types so it can be tested on the host.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace CopaOrangeBall
{
  struct Params
  {
    int step = 2;                 /**< Sampling step in visual pixels, both axes. */
    int minY = 40;                /**< Darker samples are shadow or noise. */
    int maxY = 250;               /**< Brighter samples are saturated highlights. */
    int minCr = 18;               /**< Minimum Cr - 128 of an orange sample. */
    int maxCb = -4;               /**< Maximum Cb - 128 (orange is below the gray axis). */
    int minCrMinusCb = 50;        /**< Absolute minimum Cr - Cb (lab wood: median 7-20, 99.9 % <= 39). */
    int floorMargin = 35;         /**< Cr - Cb must exceed the image median by this much (varnished wood). */
    float minHueRatio = 0.35f;    /**< (Cr - 128) / (128 - Cb): below is yellow... */
    float maxHueRatio = 1.9f;     /**< ...above is red (Trionda red stripe measured 2.19 on ospinao). */
    int minSamples = 5;           /**< Smaller blobs are noise (a ball at ~5 m covers ~12 samples at step 2). */
    float minFill = 0.42f;        /**< Blob samples / box samples; a disc fills 0.785, shading loses some. */
    float maxAspect = 1.8f;       /**< Box width / height (or inverse) for blobs not cut by the border. */
    float maxRadiusFraction = 0.45f; /**< Larger blobs (relative to the image height) are not the ball. */
    float sizeTolerance = 2.f;    /**< Accepted measured / expected radius in [1/t, t] (the projection is biased). */
    float minConfidence = 0.35f;  /**< Candidates below are dropped. */
  };

  struct Candidate
  {
    float x = 0.f;           /**< Center in visual pixels. */
    float y = 0.f;
    float radius = 0.f;      /**< Pixels. */
    float confidence = 0.f;  /**< 0..1 from color strength, roundness and size match. */
    float sizeRatio = 0.f;   /**< Measured / expected radius (0 if unknown). */
    int samples = 0;
  };

  struct Result
  {
    std::vector<Candidate> candidates; /**< Best first. */
    int floorCrMinusCb = 0;            /**< Median Cr - Cb of the image (the floor). */
    int threshold = 0;                 /**< Cr - Cb used this frame. */
    int orangeSamples = 0;
    int blobs = 0;                     /**< Blobs before the shape and size tests. */
  };

  /** Color test of one sample, given the Cr - Cb threshold of this frame. */
  inline bool isOrange(int y, int cb, int cr, int threshold, const Params& p)
  {
    const int u = cb - 128, v = cr - 128;
    if(y < p.minY || y > p.maxY || v < p.minCr || u > p.maxCb || cr - cb < threshold)
      return false;
    const float ratio = static_cast<float>(v) / static_cast<float>(-u);
    return ratio >= p.minHueRatio && ratio <= p.maxHueRatio;
  }

  /**
   * @param pixel callable (x, y, int& Y, int& Cb, int& Cr) in visual pixels.
   * @param expectedRadius callable (x, y) -> radius in pixels a ball centered there has, <= 0 if it cannot be on the floor.
   */
  template<typename Pixel, typename Expected>
  Result detect(int width, int height, Pixel pixel, Expected expectedRadius, const Params& p)
  {
    Result result;
    const int step = std::max(1, p.step);
    const int gw = width / step, gh = height / step;
    if(gw <= 0 || gh <= 0)
      return result;

    std::vector<int16_t> diff(static_cast<std::size_t>(gw * gh));
    std::vector<uint8_t> chroma(static_cast<std::size_t>(gw * gh));
    std::vector<int> histogram(511, 0);
    for(int gy = 0; gy < gh; ++gy)
      for(int gx = 0; gx < gw; ++gx)
      {
        int Y, cb, cr;
        pixel(gx * step, gy * step, Y, cb, cr);
        const int i = gy * gw + gx;
        diff[i] = static_cast<int16_t>(cr - cb);
        ++histogram[cr - cb + 255];
        // Store whether the sample passes everything except the frame threshold.
        chroma[i] = isOrange(Y, cb, cr, -255, p);
      }

    int seen = 0;
    const int half = gw * gh / 2;
    for(int d = 0; d < 511; ++d)
      if((seen += histogram[d]) > half)
      {
        result.floorCrMinusCb = d - 255;
        break;
      }
    result.threshold = std::max(p.minCrMinusCb, result.floorCrMinusCb + p.floorMargin);

    std::vector<uint8_t> mask(chroma.size());
    for(std::size_t i = 0; i < mask.size(); ++i)
    {
      mask[i] = chroma[i] && diff[i] >= result.threshold;
      result.orangeSamples += mask[i];
    }

    // 8-connected blobs on the sample grid.
    std::vector<int> stack;
    for(int start = 0; start < gw * gh; ++start)
    {
      if(!mask[start])
        continue;
      mask[start] = 0;
      stack.assign(1, start);
      int count = 0, left = gw, right = -1, top = gh, bottom = -1;
      long strength = 0;
      while(!stack.empty())
      {
        const int i = stack.back();
        stack.pop_back();
        const int gx = i % gw, gy = i / gw;
        ++count;
        strength += diff[i] - result.threshold;
        left = std::min(left, gx);
        right = std::max(right, gx);
        top = std::min(top, gy);
        bottom = std::max(bottom, gy);
        for(int dy = -1; dy <= 1; ++dy)
          for(int dx = -1; dx <= 1; ++dx)
          {
            const int nx = gx + dx, ny = gy + dy;
            if(nx < 0 || ny < 0 || nx >= gw || ny >= gh)
              continue;
            const int n = ny * gw + nx;
            if(mask[n])
            {
              mask[n] = 0;
              stack.push_back(n);
            }
          }
      }
      ++result.blobs;
      if(count < p.minSamples)
        continue;

      const int bw = right - left + 1, bh = bottom - top + 1;
      const bool cutX = left == 0 || right == gw - 1, cutY = top == 0 || bottom == gh - 1;
      const float aspect = static_cast<float>(std::max(bw, bh)) / static_cast<float>(std::min(bw, bh));
      if(!cutX && !cutY && aspect > p.maxAspect)
        continue;

      // A ball cut by the image border keeps its larger extent as diameter.
      const float radius = 0.5f * static_cast<float>(std::max(bw, bh) * step);
      if(radius > p.maxRadiusFraction * static_cast<float>(height))
        continue;
      float cx = (static_cast<float>(left + right + 1) * 0.5f) * step;
      float cy = (static_cast<float>(top + bottom + 1) * 0.5f) * step;
      if(cutX && bw < bh)
        cx = left == 0 ? (right + 1) * step - radius : left * step + radius;
      if(cutY && bh < bw)
        cy = top == 0 ? (bottom + 1) * step - radius : top * step + radius;

      // Fill relative to the full disc, so a blob cut by the border is not penalised twice.
      const float discSamples = 3.14159265f * (radius / step) * (radius / step);
      const float boxFill = static_cast<float>(count) / static_cast<float>(bw * bh);
      const float fill = cutX || cutY ? std::max(boxFill, count / discSamples) : boxFill;
      if(fill < p.minFill)
        continue;

      const float expected = expectedRadius(cx, cy);
      if(expected <= 0.f)
        continue;
      const float ratio = radius / expected;
      const float tolerance = std::max(1.01f, p.sizeTolerance);
      if(ratio > tolerance || ratio < 1.f / tolerance)
        continue;

      const float colorScore = std::min(1.f, static_cast<float>(strength) / count / 40.f);
      const float shapeScore = std::min(1.f, fill / 0.7f);
      const float sizeScore = 1.f - std::abs(std::log(ratio)) / std::log(tolerance);
      const float confidence = 0.3f + 0.7f * (0.4f + 0.6f * colorScore) * shapeScore * (0.5f + 0.5f * sizeScore);
      if(confidence < p.minConfidence)
        continue;

      Candidate& c = result.candidates.emplace_back();
      c.x = cx;
      c.y = cy;
      c.radius = radius;
      c.confidence = confidence;
      c.sizeRatio = ratio;
      c.samples = count;
    }

    std::sort(result.candidates.begin(), result.candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.confidence > b.confidence; });
    return result;
  }
}
