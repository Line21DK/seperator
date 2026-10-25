"""Convert between PNG/JPEG and PFM for sep_cli.

    python3 tools/pfm.py to-pfm in.jpg out.pfm
    python3 tools/pfm.py to-png in.pfm out.png
"""
import sys

import numpy as np
from PIL import Image


def to_pfm(src, dst):
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    with open(dst, "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (a.shape[1], a.shape[0]))
        f.write(np.ascontiguousarray(a[::-1]).astype("<f4").tobytes())


def to_png(src, dst):
    with open(src, "rb") as f:
        magic = f.readline().strip()
        w, h = map(int, f.readline().split())
        f.readline()
        c = 3 if magic == b"PF" else 1
        a = np.frombuffer(f.read(), dtype="<f4").reshape(h, w, c)[::-1]
    a = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(a.squeeze()).save(dst)


if __name__ == "__main__":
    {"to-pfm": to_pfm, "to-png": to_png}[sys.argv[1]](sys.argv[2], sys.argv[3])
