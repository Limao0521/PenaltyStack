/**
 * @file FieldColorNormalizer.h
 *
 * Adjusts the hue/saturation/value of field-green pixels before YOLO inference.
 * This is an input-domain adaptation step: the current turf color is shifted
 * toward the turf color distribution seen in the older training images.
 */

#pragma once

#include <algorithm>
#include <cmath>

namespace FieldColorNormalizer
{
struct Parameters
{
  bool enabled = true;
  float sourceHueMinDeg = 84.f;
  float sourceHueMaxDeg = 162.f;
  float minSaturation = 0.14f;
  float minValue = 0.12f;
  float greenMargin = 0.02f;
  float hueShiftDeg = -22.f;
  float saturationScale = 0.69f;
  float valueScale = 0.87f;
  float blend = 1.f;
};

inline float clamp01(const float v)
{
  return std::max(0.f, std::min(1.f, v));
}

inline float wrapHue(float hueDeg)
{
  while(hueDeg < 0.f)
    hueDeg += 360.f;
  while(hueDeg >= 360.f)
    hueDeg -= 360.f;
  return hueDeg;
}

inline bool hueInside(const float hueDeg, const float minDeg, const float maxDeg)
{
  return minDeg <= maxDeg ? (hueDeg >= minDeg && hueDeg <= maxDeg)
                          : (hueDeg >= minDeg || hueDeg <= maxDeg);
}

inline void rgbToHsv(const float r, const float g, const float b, float& h, float& s, float& v)
{
  const float maxC = std::max({r, g, b});
  const float minC = std::min({r, g, b});
  const float delta = maxC - minC;
  v = maxC;
  s = maxC > 0.f ? delta / maxC : 0.f;

  if(delta <= 1e-6f)
  {
    h = 0.f;
    return;
  }

  if(maxC == r)
    h = 60.f * std::fmod(((g - b) / delta), 6.f);
  else if(maxC == g)
    h = 60.f * (((b - r) / delta) + 2.f);
  else
    h = 60.f * (((r - g) / delta) + 4.f);
  h = wrapHue(h);
}

inline void hsvToRgb(const float hDeg, const float s, const float v, float& r, float& g, float& b)
{
  const float h = wrapHue(hDeg);
  const float c = v * s;
  const float x = c * (1.f - std::fabs(std::fmod(h / 60.f, 2.f) - 1.f));
  const float m = v - c;

  float rp = 0.f;
  float gp = 0.f;
  float bp = 0.f;
  if(h < 60.f)
  {
    rp = c;
    gp = x;
  }
  else if(h < 120.f)
  {
    rp = x;
    gp = c;
  }
  else if(h < 180.f)
  {
    gp = c;
    bp = x;
  }
  else if(h < 240.f)
  {
    gp = x;
    bp = c;
  }
  else if(h < 300.f)
  {
    rp = x;
    bp = c;
  }
  else
  {
    rp = c;
    bp = x;
  }

  r = clamp01(rp + m);
  g = clamp01(gp + m);
  b = clamp01(bp + m);
}

inline bool isSourceFieldGreen(const float r, const float g, const float b, const float h, const float s, const float v, const Parameters& params)
{
  return hueInside(h, params.sourceHueMinDeg, params.sourceHueMaxDeg) &&
         s >= params.minSaturation &&
         v >= params.minValue &&
         g >= r + params.greenMargin &&
         g >= b + params.greenMargin;
}

inline void apply(float& r, float& g, float& b, const Parameters& params)
{
  if(!params.enabled)
    return;

  float h = 0.f;
  float s = 0.f;
  float v = 0.f;
  rgbToHsv(r, g, b, h, s, v);
  if(!isSourceFieldGreen(r, g, b, h, s, v, params))
    return;

  const float blend = clamp01(params.blend);
  const float targetS = clamp01(s * params.saturationScale);
  const float targetV = clamp01(v * params.valueScale);

  h = wrapHue(h + params.hueShiftDeg * blend);
  s = clamp01(s + (targetS - s) * blend);
  v = clamp01(v + (targetV - v) * blend);
  hsvToRgb(h, s, v, r, g, b);
}
}
