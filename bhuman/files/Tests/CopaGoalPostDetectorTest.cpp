#include "../Src/Modules/Perception/CopaPerceptors/CopaGoalPostDetector.h"
#include <cassert>
#include <cmath>
#include <vector>

namespace
{
  struct Image
  {
    int w, h;
    std::vector<int> y, cb, cr;
    Image(int w, int h) : w(w), h(h), y(w * h, 150), cb(w * h, 110), cr(w * h, 140) {} // wood floor
    void fill(int x0, int y0, int x1, int y1, int Y, int Cb, int Cr)
    {
      for(int yy = y0; yy < y1; ++yy)
        for(int xx = x0; xx < x1; ++xx)
        {
          y[yy * w + xx] = Y;
          cb[yy * w + xx] = Cb;
          cr[yy * w + xx] = Cr;
        }
    }
    void blue(int x0, int y0, int x1, int y1) { fill(x0, y0, x1, y1, 45, 194, 96); } // live post pixel
    std::vector<CopaGoalPosts::Box> detect(int limit = 1 << 20) const
    {
      auto px = [this](int x, int yy, int& Y, int& Cb, int& Cr) { Y = y[yy * w + x]; Cb = cb[yy * w + x]; Cr = cr[yy * w + x]; };
      return CopaGoalPosts::detect(w, h, px, [limit](int) { return limit; }, CopaGoalPosts::Params());
    }
  };
}

int main()
{
  {
    Image image(640, 480); // goal seen from the penalty mark: two posts, crossbar, white net
    image.fill(140, 120, 500, 330, 220, 128, 128);
    image.blue(130, 110, 510, 124);
    image.blue(130, 110, 144, 330);
    image.blue(496, 110, 510, 330);
    const auto posts = image.detect();
    assert(posts.size() == 2);
    assert(posts[0].left <= 132 && posts[0].right >= 142 && posts[0].bottom >= 328);
    assert(posts[1].left <= 498 && posts[1].right >= 508);
    assert(image.detect(200).size() == 0 || image.detect(200)[0].bottom <= 200); // own body clips the scan
  }
  {
    Image image(640, 480); // blue jersey of the keeper: wide, not a post
    image.blue(280, 200, 360, 280);
    assert(image.detect().empty());
  }
  {
    Image image(320, 240); // blue patch of the Trionda: too short
    image.blue(100, 110, 112, 140);
    assert(image.detect().empty());
  }
  {
    Image image(640, 480); // close crossbar only, thick in the image
    image.blue(0, 40, 600, 100);
    assert(image.detect().empty());
  }
  assert(Image(320, 240).detect().empty());

  const float positions[3][2] = {{2000.f, 760.f}, {2050.f, -740.f}, {900.f, 100.f}};
  auto distance = [&](int a, int b) { return std::hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]); };
  const auto goal = CopaGoalPosts::pairGoal(3, distance, 1500.f, 0.25f);
  assert(goal.first == 0 && goal.second == 1);
  assert(CopaGoalPosts::pairGoal(1, distance, 1500.f, 0.25f).first == -1);
}
