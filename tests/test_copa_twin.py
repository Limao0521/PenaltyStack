"""Digital twin of the Copa behavior: protocol, map, recording and replay."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("copa_twin", ROOT / "tools/copa_twin.py")
twin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twin)
view_spec = importlib.util.spec_from_file_location("viewer_for_twin", ROOT / "tests/test_copa_goal_view.py")
view_test = importlib.util.module_from_spec(view_spec)
view_spec.loader.exec_module(view_test)


class TwinTests(unittest.TestCase):
    def test_layout_matches_the_robot(self):
        # CopaCameraStreamer.cpp writes 11 header + 13 pose + 4*9 + 1 + 2*9 + 1 + 2*13 + 12 + 5 bytes.
        self.assertEqual(twin.TWIN_FORMAT.size, 123)

    def test_parse_and_draw(self):
        data = twin.parse(view_test.twin_bytes(clearance=25.0))
        self.assertEqual((data['run'], data['state'], data['phase'], data['side']), (1, 1, 2, 1))
        self.assertTrue(data['left_post']['valid'] and data['mark']['valid'] and not data['keeper']['valid'])
        image = twin.TwinMap().draw(data)
        self.assertEqual(image.shape[0], 520)
        self.assertGreater(int(image.sum()), 0)

    def test_recorder_writes_one_file_per_run_and_replay_summarises(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = twin.TwinRecorder(Path(tmp))
            for run, phase, x, state in ((1, 0, -10.0, 1), (1, 2, -40.0, 1), (1, 3, -20.0, 1),
                                         (1, 0, -5.0, 2), (2, 0, 0.0, 1)):
                frame = twin.parse(view_test.twin_bytes(run=run, phase=phase, robot_x=x, state=state))
                frame['time'] += int(x * -10)  # distinct timestamps
                recorder.add(frame)
            recorder.close()
            files = sorted(Path(tmp).glob('*.jsonl'))
            self.assertEqual(len(files), 2)
            frames = twin.load(files[0])
            self.assertEqual(len(frames), 3)
            result = twin.summary(frames)
            self.assertEqual(result['side'], 1)
            self.assertEqual(result['posts_matched'], 3)
            self.assertAlmostEqual(result['heading_correction_deg'], 4.6, places=1)
            self.assertEqual(result['ball_moved_before_kick_mm'], 4.0)
            video = Path(tmp) / 'run.mp4'
            twin.replay(files[0], video, 1.0)
            self.assertGreater(video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
