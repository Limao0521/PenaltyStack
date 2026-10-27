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


def camf_frame(jpeg: bytes, copa: bytes | None) -> bytes:
    import struct
    body = viewer.MAGIC + struct.pack('<I', len(jpeg)) + jpeg
    body += struct.pack('<Bfff', 1, 100.0, 120.0, 20.0)   # ball percept
    body += bytes([1]) + struct.pack('<ii', 5, 6)           # one ball spot
    body += bytes([0])                                      # no patch
    return body + (copa or b'')


def copa_section() -> bytes:
    import struct
    box = struct.Struct('<hhhhBfff')
    out = viewer.COPA_MAGIC + bytes([3]) + struct.pack('<Bff', 1, 450.0, 20.0)
    out += bytes([2]) + box.pack(30, 40, 10, 200, 3, 2000.0, 760.0, 0.46) + box.pack(280, 290, 12, 205, 3, 2050.0, -740.0, -0.46)
    out += struct.pack('<Bbbfff', 1, 0, 1, 0.92, 1500.0, 0.0)
    out += bytes([1]) + box.pack(140, 180, 60, 190, 3, 1800.0, 0.0, 0.05) + bytes([0])
    return out + bytes([1]) + twin_bytes()


def twin_bytes(run=1, phase=2, robot_x=-40.0, clearance=55.0) -> bytes:
    return viewer.copa_twin.TWIN_FORMAT.pack(
        12000, run, 0, 1, phase,
        1, robot_x, -20.0, 0.05,                          # robot pose
        1, 210.0, 0.0, 1, 200.0, 0.0,                     # ball, mark
        1, 1500.0, 760.0, 1, 1520.0, -740.0, 0,           # posts, not inferred
        0, 0.0, 0.0, 1, 1500.0, 380.0, 1,                 # keeper, target, side left
        1, -90.0, -60.0, 0.02, 1, -90.0, -60.0, 0.02,     # pre-kick, walk target
        clearance, 48.0, 4.0)


class CopaStreamTests(unittest.TestCase):
    def setUp(self):
        import socket
        self.robot, self.pc = socket.socketpair()
        self.jpeg = cv2.imencode('.jpg', np.zeros((240, 320, 3), np.uint8))[1].tobytes()

    def tearDown(self):
        self.robot.close()
        self.pc.close()

    def test_copa_section_is_parsed(self):
        self.robot.sendall(camf_frame(self.jpeg, copa_section()) + camf_frame(self.jpeg, copa_section()))
        for _ in range(2):
            jpeg, status, bx, by, br, spots, copa = viewer.read_frame(self.pc)
            self.assertEqual((status, spots), (1, [(5, 6)]))
            self.assertEqual(len(copa['posts']), 2)
            self.assertTrue(copa['goal']['seen'])
            self.assertAlmostEqual(copa['goal']['span'], 0.92, places=5)
            self.assertAlmostEqual(copa['posts'][0]['bearing'], 0.46, places=5)
            self.assertEqual(len(copa['robots']), 1)
            self.assertTrue(copa['posts'][1]['on_field'])
            self.assertEqual(copa['twin']['phase'], 2)
            self.assertAlmostEqual(copa['twin']['target']['y'], 380.0)

    def test_plain_camf_frames_stay_in_sync(self):
        self.robot.sendall(camf_frame(self.jpeg, None) * 3)
        for _ in range(2):  # the third frame has no follower to peek at
            jpeg, status, *_, copa = viewer.read_frame(self.pc)
            self.assertEqual(jpeg, self.jpeg)
            self.assertIsNone(copa)

    def test_copa_overlay_reports_ball_goal_and_robot(self):
        self.robot.sendall(camf_frame(self.jpeg, copa_section()) + viewer.MAGIC)
        *_, copa = viewer.read_frame(self.pc)
        vis = np.zeros((480, 640, 3), np.uint8)
        lines = [text for text, _ in viewer.draw_copa(vis, copa, 2, viewer.SEEN)]
        self.assertTrue(lines[0].startswith("BALON 0.45m"))
        self.assertIn("PORTERIA ok 53deg ~1.50m", lines[1])
        self.assertEqual(lines[2], "NAO 1 +3deg")
        self.assertGreater(int(vis.sum()), 0)
        viewer.draw_overlay(vis, viewer.SEEN, 100, 120, 20, [], 2, "UPPER", 30.0, 1, None, 20, None, 0.0, copa)
