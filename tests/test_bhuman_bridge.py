import importlib.util
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch


class BridgeFailureTests(unittest.TestCase):
    def bridge(self, mode):
        self.assertIsNotNone(importlib.util.find_spec('penalty.bhuman_bridge'),
                             'native process bridge is missing')
        from penalty.bhuman_bridge import BHumanBridge
        script = Path(__file__).parent/'fixtures'/'motion_process.py'
        # -m imports the fixture from the repository root, like the real launcher.
        return BHumanBridge(role='striker',trial='test',timeout=.25,
            command=[sys.executable,'-u','-m','tests.fixtures.motion_process',mode])

    def frame(self):
        return {'angles':[0.]*26,'velocity':[0.]*26,'gyro':[0.]*3,
                'acc':[0,0,9.81],'orientation':[1,0,0,0],'fsr':[.7]*8}

    def test_wrong_sequence_and_nonfinite_output_never_reach_servos(self):
        for mode in ('sequence','nonfinite'):
            with self.subTest(mode=mode):
                bridge = self.bridge(mode)
                try:
                    with self.assertRaises(RuntimeError):
                        bridge.step(self.frame(),{'motion':'stand'},sequence=0,time_ms=10000)
                finally:
                    bridge.close()
                self.assertIsNotNone(bridge.process.poll())

    def test_missing_response_has_bounded_timeout_and_collects_child(self):
        bridge = self.bridge('timeout')
        started = time.monotonic()
        try:
            with self.assertRaises(RuntimeError):
                bridge.step(self.frame(),{'motion':'stand'},sequence=0,time_ms=10000)
            self.assertLess(time.monotonic()-started,1.)
        finally:
            bridge.close()
        self.assertIsNotNone(bridge.process.poll())

    def test_shutdown_enqueue_failure_still_collects_child(self):
        bridge = self.bridge('timeout')
        started = time.monotonic()
        try:
            # A saturated outgoing queue raises from _send before stdin can
            # consume SHUTDOWN. Cleanup must still terminate the peer.
            with patch.object(bridge,'_send',side_effect=RuntimeError('full queue')):
                bridge.close()
            self.assertIsNotNone(bridge.process.poll())
            self.assertLess(time.monotonic()-started,2.)
        finally:
            if bridge.process.poll() is None:
                bridge.process.kill()
                bridge.process.wait(timeout=2)


if __name__ == '__main__':
    unittest.main()
