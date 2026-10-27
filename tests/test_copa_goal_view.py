import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("viewer", Path(__file__).resolve().parents[1] / "tools/copa_camera_viewer.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class GoalPostViewerTests(unittest.TestCase):
    def test_two_blue_vertical_posts_are_detected(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.rectangle(frame, (40, 30), (53, 180), (255, 0, 0), -1)
        cv2.rectangle(frame, (250, 30), (263, 180), (255, 0, 0), -1)
        posts = viewer.find_blue_posts(frame)
        self.assertEqual(len(posts), 2)
        self.assertLess(posts[0][0], posts[1][0])

    def test_red_and_horizontal_blue_are_not_posts(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.rectangle(frame, (20, 70), (250, 85), (255, 0, 0), -1)
        cv2.rectangle(frame, (280, 20), (295, 200), (0, 0, 255), -1)
        self.assertEqual(viewer.find_blue_posts(frame), [])


if __name__ == "__main__":
    unittest.main()
