import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_occ_heatmap(occ_path: str, reduction: str = 'mean'):
    occ = np.load(occ_path)
    if occ.ndim == 2:
        return occ
    if occ.ndim == 3:
        if occ.shape[0] == 1:
            return occ[0]
        if reduction == 'first':
            return occ[0]
        if reduction == 'max':
            return occ.max(axis=0)
        return occ.mean(axis=0)
    raise ValueError(f'Unsupported occ shape: {occ.shape} from {occ_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Render occ.npy files into heatmap PNGs.')
    parser.add_argument('--occ-dir', default='work_dirs/occ_maps', help='Directory containing *_occ.npy files.')
    parser.add_argument('--out-dir', default=None, help='Output directory for *_occ.png. Defaults to occ-dir.')
    parser.add_argument('--reduction', default='mean', choices=['mean', 'first', 'max'], help='How to reduce multi-channel occ.')
    parser.add_argument('--dpi', type=int, default=150, help='Saved image DPI.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing PNGs.')
    return parser.parse_args()


def main():
    args = parse_args()
    occ_dir = args.occ_dir
    out_dir = args.out_dir or occ_dir
    os.makedirs(out_dir, exist_ok=True)

    occ_files = sorted(
        os.path.join(occ_dir, name)
        for name in os.listdir(occ_dir)
        if name.endswith('_occ.npy')
    )
    if not occ_files:
        raise RuntimeError(f'No *_occ.npy files found in {occ_dir}')

    for occ_path in occ_files:
        base = os.path.basename(occ_path)
        out_path = os.path.join(out_dir, base[:-4] + '.png')
        if os.path.isfile(out_path) and not args.overwrite:
            continue

        heatmap = _load_occ_heatmap(occ_path, reduction=args.reduction)
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, origin='lower', cmap='hot')
        plt.colorbar()
        plt.title(base)
        plt.tight_layout()
        plt.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f'[occ-render] saved {out_path}')


if __name__ == '__main__':
    main()
