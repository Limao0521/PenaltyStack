"""Regenerate the no-shot posture contact sheet from physical render frames."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parent


def font(size):
    for name in ('C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


canvas = Image.new('RGB', (900, 770), '#f5f6f8')
draw = ImageDraw.Draw(canvas)
draw.text((15, 10), 'Comparación de postura en MuJoCo — sin tiro', font=font(20), fill='#22272d')
for x, label in ((15, 'Guardia original'), (465, 'Rodillas +4.6° (descartada)')):
    draw.text((x, 47), label, font=font(20), fill='#22272d')
for row, time_s in enumerate((0.72, 1.20)):
    top = 82 + row * 344
    for column, profile in enumerate(('nominal', 'knee')):
        frame = Image.open(root / profile / ('center_%.2fs.png' % time_s)).convert('RGB')
        canvas.paste(frame.resize((450, 300), Image.Resampling.LANCZOS), (450 * column, top))
        draw.text((15 + 450 * column, top + 310),
                  '%.2f s desde solicitar la guardia' % time_s, font=font(16), fill='#22272d')
canvas.save(root / 'posture_comparison.png')
