"""Run the original robot network on a local, unmirrored VGA/QVGA image."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.robot_detector import RobotDetector


def main():
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    with Image.open(args.image) as image:
        rgb = image.convert('RGB')
    detector = RobotDetector()
    try:
        output = {'input': str(Path(args.image).resolve()), 'runtime': detector.hello,
                  'detections': detector.detect(rgb.tobytes(), *rgb.size, 0.),
                  'ground_position': 'requires calibrated camera transform; not inferred from a photo',
                  'opponent_yaw_valid': False}
    finally:
        detector.close()
    text = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text+'\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
