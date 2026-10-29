#include "../Src/Modules/Perception/BallPerceptors/CopaOrangeBall.h"
#include <cassert>
#include <cmath>
#include <vector>

namespace
{
  struct Color { int y, cb, cr; };
  constexpr Color labWood{126, 118, 125};       // ospinao lower camera, 2026-10-27 (Cr - Cb median 7)
  constexpr Color varnishedWood{150, 95, 154};  // a more orange duela (Cr - Cb 59)
  constexpr Color orange{130, 78, 185};         // orange plastic (Cr - Cb 107)
  constexpr Color shadedOrange{60, 100, 160};   // dark side of the ball
  constexpr Color skin{160, 110, 150};
  constexpr Color red{80, 110, 190};

  struct Image
  {
    int w, h;
    std::vector<Color> px;
    Image(int w, int h, Color floor) : w(w), h(h), px(w * h, floor)
    {
      for(int i = 0; i < w * h; ++i) // wood grain
        px[i].cr += (i * 7919 % 13) - 6;
    }
    void disc(float cx, float cy, float r, Color c)
    {
      for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
          if(std::hypot(x - cx, y - cy) <= r)
            px[y * w + x] = c;
    }
    void box(int x0, int y0, int x1, int y1, Color c)
    {
      for(int y = y0; y < y1; ++y)
        for(int x = x0; x < x1; ++x)
          px[y * w + x] = c;
    }
    CopaOrangeBall::Result detect(float expected) const
    {
      auto pixel = [this](int x, int y, int& Y, int& cb, int& cr) { const Color& c = px[y * w + x]; Y = c.y; cb = c.cb; cr = c.cr; };
      return CopaOrangeBall::detect(w, h, pixel, [expected](float, float) { return expected; }, CopaOrangeBall::Params());
    }
  };

  bool near(float a, float b, float tolerance) { return std::abs(a - b) <= tolerance; }
}

int main()
{
  for(const Color floor : {labWood, varnishedWood})
  {
    Image image(320, 240, floor); // lower camera, ball in front of the feet
    image.disc(160, 120, 20, orange);
    const auto result = image.detect(20.f);
    assert(result.candidates.size() == 1);
    const auto& ball = result.candidates.front();
    assert(near(ball.x, 160, 2) && near(ball.y, 120, 2) && near(ball.radius, 20, 3));
    assert(ball.confidence > 0.6f);
    assert(image.detect(60.f).candidates.empty()); // three times smaller than a ball there
    assert(image.detect(0.f).candidates.empty());  // above the horizon
  }
  assert(Image(320, 240, varnishedWood).detect(20.f).candidates.empty());

  {
    Image image(320, 240, labWood); // shaded lower half and a white highlight
    image.disc(150, 100, 24, orange);
    for(int y = 108; y < 125; ++y)
      for(int x = 126; x < 175; ++x)
        if(std::hypot(x - 150, y - 100) <= 24)
          image.px[y * 320 + x] = shadedOrange;
    image.disc(143, 92, 5, Color{250, 128, 128});
    const auto result = image.detect(24.f);
    assert(result.candidates.size() == 1 && near(result.candidates.front().x, 150, 3));
  }
  {
    Image image(320, 240, labWood); // ball cut by the bottom border, close to the toes
    image.disc(160, 228, 40, orange);
    const auto result = image.detect(40.f);
    assert(result.candidates.size() == 1);
    assert(near(result.candidates.front().y, 228, 4) && near(result.candidates.front().radius, 40, 4));
  }
  {
    Image image(640, 480, labWood); // upper camera, ball about 4 m away
    image.disc(400, 200, 6, orange);
    const auto result = image.detect(6.f);
    assert(result.candidates.size() == 1 && near(result.candidates.front().x, 400, 2));
  }
  {
    Image image(320, 240, labWood);
    image.disc(100, 120, 22, skin);           // hand
    image.disc(220, 120, 22, red);            // red jersey / Trionda patch
    image.box(20, 200, 300, 212, orange);     // orange cable or tape
    assert(image.detect(22.f).candidates.empty());
  }
  {
    Image image(320, 240, labWood); // two balls: both reported, best first
    image.disc(80, 120, 18, orange);
    image.disc(240, 120, 18, shadedOrange);
    image.disc(240, 120, 18, Color{140, 82, 170});
    const auto result = image.detect(18.f);
    assert(result.candidates.size() == 2 && result.candidates[0].confidence >= result.candidates[1].confidence);
  }
  {
    Image image(320, 240, labWood);
    const auto result = image.detect(20.f);
    assert(result.candidates.empty() && near(static_cast<float>(result.floorCrMinusCb), 7, 3));
  }
}
