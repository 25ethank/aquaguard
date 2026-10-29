"""Turn a slice of the pipeline's demo video into a README GIF (no ffmpeg needed).

  python make_gif.py demo.mp4 demo.gif --start 5 --seconds 12
"""
import argparse
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('out')
ap.add_argument('--start', type=float, default=0, help='seconds into the video')
ap.add_argument('--seconds', type=float, default=12)
ap.add_argument('--fps', type=float, default=10)
ap.add_argument('--width', type=int, default=640)
a = ap.parse_args()

cap = cv2.VideoCapture(a.src)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * src_fps))
step = max(1, round(src_fps / a.fps))

frames = []
for i in range(int(a.seconds * src_fps)):
    ok, f = cap.read()
    if not ok:
        break
    if i % step:
        continue
    h, w = f.shape[:2]
    f = cv2.resize(f, (a.width, int(h * a.width / w)), interpolation=cv2.INTER_AREA)
    frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                  .quantize(colors=128, method=Image.Quantize.MEDIANCUT))
cap.release()

frames[0].save(a.out, save_all=True, append_images=frames[1:],
               duration=int(1000 * step / src_fps), loop=0, optimize=True)
print(f"Wrote {a.out}: {len(frames)} frames")
