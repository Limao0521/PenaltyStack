"""Visual body yaw from rendered NAO references, never opponent scene state.

This v1 estimator is restricted to the MuJoCo NAO V6 appearance domain.
Reference poses are offline labeled renders, not trajectories in a penalty.
It is not a validated physical-NAO vision model. NumPy/Pillow, Python 3.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

VERSION = 'nao_visual_yaw_v1'
DEFAULT_MODEL = Path(__file__).resolve().parents[1]/'models/vision/nao_yaw_v1'
FEATURE_SIZE = (32, 48)


def wrap(angle):
    return (float(angle)+math.pi) % (2*math.pi)-math.pi


def body_feature(rgb, box):
    """Pixel-only neutral body crop; no segmentation IDs or depth buffer."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('uint8 RGB camera image required')
    bounds = np.asarray(box, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError('finite robot box required')
    left, top = np.floor(bounds[:2]).astype(int)
    right, bottom = np.ceil(bounds[2:]).astype(int)
    if left < 0 or top < 0 or right >= rgb.shape[1] or bottom >= rgb.shape[0]:
        return None, {'reason': 'clipped_body'}
    margin = max(4, round((bottom-top)*.06))
    left, top = max(0, left-margin), max(0, top-margin)
    right, bottom = min(rgb.shape[1]-1, right+margin), min(rgb.shape[0]-1, bottom+margin)
    patch = rgb[top:bottom+1, left:right+1]
    if patch.shape[0] < 40 or patch.shape[1] < 12:
        return None, {'reason': 'too_few_pixels'}
    high, low = patch.max(2).astype(float), patch.min(2).astype(float)
    gray = patch.mean(2)
    neutral = (high-low <= .18*high+8) & (gray > 38)
    yy, xx = np.nonzero(neutral)
    if len(xx) < 120:
        return None, {'reason': 'body_not_visible'}
    x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
    height, width = y1-y0+1, x1-x0+1
    if height < 40 or not 1.45 <= height/max(1, width) <= 4.8:
        return None, {'reason': 'incomplete_or_nonstanding_body'}
    if y0 == 0 or y1 == patch.shape[0]-1:
        return None, {'reason': 'body_touches_crop_edge'}
    # Keep body aspect ratio. Background, orange ball and green shadows are
    # removed by color only. Dark body details remain as holes in the mask.
    g = np.where(neutral, gray, 0)[y0:y1+1, x0:x1+1].astype(np.uint8)
    mask = (neutral[y0:y1+1, x0:x1+1]*255).astype(np.uint8)
    out_w, out_h = FEATURE_SIZE
    scale = min((out_w-2)/width, (out_h-2)/height)
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    channels = []
    for source in (g, mask):
        small = np.asarray(Image.fromarray(source).resize(size, Image.Resampling.BILINEAR))
        canvas = np.zeros((out_h, out_w), dtype=np.float32)
        dy, dx = (out_h-size[1])//2, (out_w-size[0])//2
        canvas[dy:dy+size[1], dx:dx+size[0]] = small/255.
        # A turned head must not be interpreted as a turned torso.
        canvas[:12] *= .25
        channels.append(canvas)
    feature = np.stack(channels).ravel()
    return feature, {'reason': 'visible_body', 'body_box': [int(left+x0), int(top+y0),
                       int(left+x1), int(top+y1)], 'body_pixels': int(len(xx)),
                     'body_height_px': int(height)}


class VisualYawEstimator:
    def __init__(self, model=DEFAULT_MODEL):
        model = Path(model)
        self.metadata = json.loads((model/'manifest.json').read_text(encoding='utf-8'))
        raw = (model/'templates.npz').read_bytes()
        if (self.metadata['version'] != VERSION or
                hashlib.sha256(raw).hexdigest() != self.metadata['templates_sha256']):
            raise ValueError('orientation template version/hash differs')
        with np.load(model/'templates.npz', allow_pickle=False) as data:
            self.features = data['features'].astype(np.float32)/255.
            self.yaws = data['yaws'].astype(np.float64)
        if (self.features.ndim != 2 or self.features.shape[1] != 2*32*48 or
                self.yaws.shape != (len(self.features),) or len(self.yaws) < 24 or
                not np.isfinite(self.features).all() or not np.isfinite(self.yaws).all()):
            raise ValueError('invalid visual reference bank')
        self.norms = np.einsum('ij,ij->i', self.features, self.features)
        self.reset()

    def reset(self):
        self.previous = None
        self.last_capture = None

    def estimate(self, rgb, box, capture_s, decision_s, track_id, bearing,
                 own_yaw=0., synchronized=True, source='simulated_camera', odometry_yaw=0.):
        """Yaw in current own ground frame; ambiguity/misses clear validity.

        bearing is measured robot bearing at capture, own_yaw compensates
        capture-to-decision own rotation. No rival velocities are inputs.
        """
        start = time.perf_counter()
        if not all(math.isfinite(float(v)) for v in (capture_s, decision_s, bearing, own_yaw, odometry_yaw)):
            raise ValueError('finite capture/decision/bearing/own rotation required')
        result = {'version': VERSION, 'yaw': 0., 'yaw_valid': False,
                  'yaw_variance': math.pi**2, 'capture_s': float(capture_s),
                  'age_s': float(decision_s-capture_s), 'track_id': track_id,
                  'model_sha256': self.metadata['templates_sha256'],
                  'domain': 'simulated_nao_v6_rgb', 'hardware_validated': False}
        def finish(reason):
            result['reason'] = reason
            result['processing_ms'] = (time.perf_counter()-start)*1000
            if not result['yaw_valid'] and reason != 'confirming_orientation':
                self.previous = None
            return result
        if source != 'simulated_camera':
            return finish('appearance_domain_not_validated')
        if not synchronized or not 0 <= decision_s-capture_s <= .25:
            return finish('stale_or_unsynchronized')
        if self.last_capture is not None and capture_s <= self.last_capture:
            return finish('duplicate_or_out_of_order')
        self.last_capture = float(capture_s)
        feature, detail = body_feature(rgb, box)
        result.update(detail)
        if feature is None:
            return finish(detail['reason'])
        # Match appearance; circular neighbors preserve the -pi/+pi seam.
        distance = np.maximum(0., self.norms + feature@feature - 2*(self.features@feature))/len(feature)
        indices = np.argsort(distance)[:12]
        best = float(distance[indices[0]])
        weights = np.exp(-(distance[indices]-best)/.002)
        weights /= weights.sum()
        resultant = np.sum(weights*np.exp(1j*self.yaws[indices]))
        length = float(abs(resultant))
        sigma = math.sqrt(max(0., -2*math.log(max(length, 1e-9))))
        candidate = wrap(float(np.angle(resultant))+bearing+own_yaw)
        result.update(candidate_yaw=candidate, match_distance=best,
                      circular_concentration=length, angular_sigma_rad=sigma,
                      yaw_variance=max(math.radians(7.5)**2, sigma**2))
        if best > self.metadata['max_match_distance']:
            return finish('unfamiliar_appearance')
        if sigma > self.metadata['max_angular_sigma_rad']:
            return finish('ambiguous_body_heading')
        previous = self.previous
        world_candidate = wrap(candidate+odometry_yaw)
        self.previous = (float(capture_s), track_id, world_candidate)
        if (previous is None or track_id != previous[1] or not 0 < capture_s-previous[0] <= .25 or
                abs(wrap(world_candidate-previous[2])) > .45):
            return finish('confirming_orientation')
        result.update(yaw=candidate, yaw_valid=True)
        return finish('measured_visual_orientation')
