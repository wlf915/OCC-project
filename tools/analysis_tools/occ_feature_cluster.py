import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def _require_sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise ImportError(
            'This script requires scikit-learn. Please install it first, e.g. `pip install scikit-learn`.'
        ) from exc
    return KMeans, TSNE, adjusted_rand_score, normalized_mutual_info_score, StandardScaler


def _load_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_run_index(run_dir):
    meta_dir = os.path.join(run_dir, 'meta')
    index = {}

    if os.path.isdir(meta_dir):
        for name in os.listdir(meta_dir):
            if not name.endswith('.jsonl'):
                continue
            for row in _load_jsonl(os.path.join(meta_dir, name)):
                sid = row.get('sample_id')
                if sid:
                    index[sid] = row

    bev_dir = os.path.join(run_dir, 'bev')
    if os.path.isdir(bev_dir):
        for name in os.listdir(bev_dir):
            if not name.endswith('_bev.npy'):
                continue
            sid = name[:-8]
            record = index.setdefault(sid, {'sample_id': sid, 'paths': {}})
            record.setdefault('paths', {})['bev'] = os.path.join(bev_dir, name)

    occ_dir = os.path.join(run_dir, 'occ')
    if os.path.isdir(occ_dir):
        for name in os.listdir(occ_dir):
            if not name.endswith('_occ.npy'):
                continue
            sid = name[:-8]
            record = index.setdefault(sid, {'sample_id': sid, 'paths': {}})
            record.setdefault('paths', {})['occ'] = os.path.join(occ_dir, name)

    cls_dir = os.path.join(run_dir, 'cls')
    if os.path.isdir(cls_dir):
        for name in os.listdir(cls_dir):
            if not name.endswith('_cls_last.npy'):
                continue
            sid = name[:-13]
            record = index.setdefault(sid, {'sample_id': sid, 'paths': {}})
            record.setdefault('paths', {})['cls_last'] = os.path.join(cls_dir, name)

    return index


def _load_bev(path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f'BEV feature should be 3D (C,H,W), got shape={arr.shape} from {path}')

    # Expected export format is (C,H,W). If accidentally saved as (H,W,C), convert.
    c0, c1, c2 = arr.shape
    if c0 < 32 and c2 >= 64:
        arr = arr.transpose(2, 0, 1)
    return arr


def _build_region_labels(occ, threshold=0.5):
    if occ.ndim == 2:
        return (occ > threshold).astype(np.int64)

    if occ.ndim == 3:
        if occ.shape[0] == 1:
            return (occ[0] > threshold).astype(np.int64)
        return np.argmax(occ, axis=0).astype(np.int64)

    raise ValueError(f'OCC map should be 2D or 3D, got shape={occ.shape}')


def _aggregate_region_features(bev_feat, region_labels):
    c, h, w = bev_feat.shape
    if region_labels.shape != (h, w):
        raise ValueError(
            f'Region label shape mismatch: labels={region_labels.shape}, bev={(h, w)}'
        )

    region_ids = np.unique(region_labels)
    result = {}
    flat = bev_feat.reshape(c, -1)
    labels_flat = region_labels.reshape(-1)

    for rid in region_ids:
        mask = labels_flat == rid
        pixel_count = int(mask.sum())
        if pixel_count <= 0:
            continue
        vec = flat[:, mask].mean(axis=1)
        result[int(rid)] = (vec, pixel_count)
    return result


def _cosine_similarity(a, b, eps=1e-8):
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 0.0
    return float(np.dot(a, b) / (an * bn + eps))


def _save_scatter(points, labels, out_path, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, s=8, cmap='tab20', alpha=0.75)
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cluster BEV regional features and compare with-occ vs no-occ experiments.'
    )
    parser.add_argument('--with-occ-dir', required=True, help='Export dir of the with-occ run.')
    parser.add_argument('--no-occ-dir', required=True, help='Export dir of the no-occ run.')
    parser.add_argument('--out-dir', required=True, help='Directory to save analysis results.')
    parser.add_argument('--k', type=int, default=8, help='Number of clusters for KMeans.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-samples', type=int, default=0, help='Max sample_token count (0 means all).')
    parser.add_argument('--occ-threshold', type=float, default=0.5, help='Threshold for binary occ map.')
    parser.add_argument('--save-tsne', action='store_true', help='If set, save t-SNE scatter plot.')
    return parser.parse_args()


def main():
    args = parse_args()
    KMeans, TSNE, adjusted_rand_score, normalized_mutual_info_score, StandardScaler = _require_sklearn()

    os.makedirs(args.out_dir, exist_ok=True)

    with_index = _load_run_index(args.with_occ_dir)
    no_index = _load_run_index(args.no_occ_dir)

    common_tokens = sorted(set(with_index.keys()) & set(no_index.keys()))
    if args.max_samples > 0:
        common_tokens = common_tokens[:args.max_samples]

    pairs = []
    skipped = defaultdict(int)

    for sid in common_tokens:
        with_paths = with_index[sid].get('paths', {})
        no_paths = no_index[sid].get('paths', {})

        bev_with_path = with_paths.get('bev')
        bev_no_path = no_paths.get('bev')
        occ_path = with_paths.get('occ')

        if not bev_with_path or not os.path.isfile(bev_with_path):
            skipped['missing_with_bev'] += 1
            continue
        if not bev_no_path or not os.path.isfile(bev_no_path):
            skipped['missing_no_bev'] += 1
            continue
        if not occ_path or not os.path.isfile(occ_path):
            skipped['missing_with_occ'] += 1
            continue

        bev_with = _load_bev(bev_with_path)
        bev_no = _load_bev(bev_no_path)
        if bev_with.shape != bev_no.shape:
            skipped['shape_mismatch_bev'] += 1
            continue

        occ = np.load(occ_path)
        labels = _build_region_labels(occ, threshold=args.occ_threshold)

        try:
            with_region = _aggregate_region_features(bev_with, labels)
            no_region = _aggregate_region_features(bev_no, labels)
        except Exception:
            skipped['shape_mismatch_occ'] += 1
            continue

        region_ids = sorted(set(with_region.keys()) & set(no_region.keys()))
        for rid in region_ids:
            vec_with, pix = with_region[rid]
            vec_no, _ = no_region[rid]
            pairs.append(
                {
                    'sample_id': sid,
                    'region_id': int(rid),
                    'pixel_count': int(pix),
                    'feat_with': vec_with,
                    'feat_no': vec_no,
                    'pair_cos': _cosine_similarity(vec_with, vec_no),
                }
            )

    if not pairs:
        raise RuntimeError('No valid paired region features found. Check export dirs and occ files.')

    x_with = np.stack([p['feat_with'] for p in pairs], axis=0)
    x_no = np.stack([p['feat_no'] for p in pairs], axis=0)
    x_all = np.concatenate([x_with, x_no], axis=0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all)
    n_with = x_with.shape[0]
    xw_scaled = x_scaled[:n_with]
    xn_scaled = x_scaled[n_with:]

    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels_with = kmeans.fit_predict(xw_scaled)
    labels_no = kmeans.predict(xn_scaled)

    ari = float(adjusted_rand_score(labels_with, labels_no))
    nmi = float(normalized_mutual_info_score(labels_with, labels_no))
    pair_cos_mean = float(np.mean([p['pair_cos'] for p in pairs]))

    # Cluster-center cosine similarity between domains under the same cluster ID.
    center_cos = []
    for cid in range(args.k):
        mw = labels_with == cid
        mn = labels_no == cid
        if mw.sum() == 0 or mn.sum() == 0:
            continue
        cw = x_with[mw].mean(axis=0)
        cn = x_no[mn].mean(axis=0)
        center_cos.append(_cosine_similarity(cw, cn))
    center_cos_mean = float(np.mean(center_cos)) if center_cos else 0.0

    metrics = {
        'num_pairs': len(pairs),
        'num_common_tokens': len(common_tokens),
        'skipped': dict(skipped),
        'k': args.k,
        'seed': args.seed,
        'ari_cluster_with_vs_no': ari,
        'nmi_cluster_with_vs_no': nmi,
        'pairwise_cosine_mean': pair_cos_mean,
        'center_cosine_mean': center_cos_mean,
    }

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out_dir, 'paired_regions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id',
            'region_id',
            'pixel_count',
            'pair_cosine',
            'cluster_with',
            'cluster_no',
        ])
        for i, p in enumerate(pairs):
            writer.writerow([
                p['sample_id'],
                p['region_id'],
                p['pixel_count'],
                f"{p['pair_cos']:.8f}",
                int(labels_with[i]),
                int(labels_no[i]),
            ])

    confusion = np.zeros((args.k, args.k), dtype=np.int64)
    for lw, ln in zip(labels_with, labels_no):
        confusion[int(lw), int(ln)] += 1
    np.save(os.path.join(args.out_dir, 'cluster_confusion.npy'), confusion)

    if args.save_tsne:
        tsne = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto')
        proj = tsne.fit_transform(np.concatenate([xw_scaled, xn_scaled], axis=0))
        domain_labels = np.concatenate([np.zeros(n_with, dtype=np.int64), np.ones(n_with, dtype=np.int64)], axis=0)
        _save_scatter(
            proj,
            domain_labels,
            os.path.join(args.out_dir, 'tsne_domain.png'),
            't-SNE domain (0=with_occ, 1=no_occ)',
        )
        cluster_labels = np.concatenate([labels_with, labels_no], axis=0)
        _save_scatter(
            proj,
            cluster_labels,
            os.path.join(args.out_dir, 'tsne_cluster.png'),
            't-SNE cluster labels',
        )

    print('[OCC-Cluster] Done.')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
