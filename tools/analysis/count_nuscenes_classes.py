#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count per-class instance numbers from a nuScenes info pkl/json under a given root.

Usage:
  python tools/analysis/count_nuscenes_classes.py --data-root /dataset \
         --info nuscenes_infos_temporal_train.pkl

It will search <data-root>/<info> and print counts.
"""
import argparse
import os
from collections import Counter

import mmcv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.getenv('NUSCENES_DATA_ROOT', 'data/nuscenes'))
    parser.add_argument('--info', type=str, default='nuscenes_infos_temporal_train.pkl')
    args = parser.parse_args()

    info_path = args.info
    if not os.path.isabs(info_path):
        info_path = os.path.join(args.data_root, info_path)
    assert os.path.exists(info_path), f'Info file not found: {info_path}'

    data = mmcv.load(info_path)
    # compatible with list[dict] or dict with key 'infos'
    infos = data.get('infos', data) if isinstance(data, dict) else data

    cnt = Counter()
    for rec in infos:
        annos = rec.get('annos') or rec.get('gt_boxes', None)
        names = None
        if isinstance(annos, dict):
            names = annos.get('names') or annos.get('gt_names')
        if names is None:
            names = rec.get('gt_names')
        if names is None:
            continue
        for n in names:
            cnt[str(n)] += 1

    total = sum(cnt.values())
    print('Total instances:', total)
    for k, v in cnt.most_common():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
