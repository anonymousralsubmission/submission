
# compute_class_weights.py
# Compute median-frequency class weights for 3-class masks (labels 0,1,2; 255=ignore).
# Usage:
#   python compute_class_weights.py --labels datasets/MyEdges/Labels --ext .png
import argparse, os, numpy as np
from PIL import Image
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='Path to Labels folder (recursively scanned)')
    ap.add_argument('--ext', default='.png', help='Mask file extension')
    args = ap.parse_args()
    counts = np.zeros(3, dtype=np.int64)
    pres   = np.zeros(3, dtype=np.int64)
    for root, _, files in os.walk(args.labels):
        for fn in files:
            if not fn.lower().endswith(args.ext):
                continue
            m = np.array(Image.open(os.path.join(root, fn)))
            for c in range(3):
                n = (m == c).sum()
                counts[c] += n
                pres[c]   += int(n > 0)
    freq = counts / np.maximum(pres, 1)
    freq = np.where(freq == 0, np.nan, freq)
    median_freq = np.nanmedian(freq)
    weights = median_freq / freq
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    print('counts:', counts.tolist())
    print('present_in:', pres.tolist())
    print('weights (median-frequency):', weights.tolist())
    print('Suggested CLASS_WEIGHT to paste in config:')
    print([round(float(w), 6) for w in weights.tolist()])
if __name__ == '__main__':
    main()
