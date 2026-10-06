import unittest

from penalty.vision import detect_orange_ball


class VisionTests(unittest.TestCase):
    def test_centered_ball(self):
        w, h = 80, 60
        pixels = bytearray([20, 20, 20] * (w * h))
        for y in range(26, 36):
            for x in range(35, 45):
                i = (y * w + x) * 3
                pixels[i:i + 3] = bytes([230, 110, 20])
        result = detect_orange_ball(pixels, w, h)
        self.assertIsNotNone(result)
        self.assertLess(abs(result[1]), 0.1)

    def test_empty(self):
        self.assertIsNone(detect_orange_ball(bytes([0, 0, 0] * 100), 10, 10))


if __name__ == '__main__':
    unittest.main()
