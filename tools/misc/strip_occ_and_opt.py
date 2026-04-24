import argparse
import torch


def strip_checkpoint(src_path: str, dst_path: str, drop_occ: bool = True, drop_optimizer: bool = True):
    ckpt = torch.load(src_path, map_location='cpu')

    # handle state_dict position
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    if drop_occ:
        keys = list(state.keys())
        removed = 0
        for k in keys:
            name = k
            # common prefixes to consider
            if name.startswith('module.'):
                name_wo_module = name[len('module.') :]
            else:
                name_wo_module = name
            if name_wo_module.startswith('pts_bbox_head.occ_head.'):
                state.pop(k, None)
                removed += 1
        print(f'Removed {removed} occ_head parameters')

    if drop_optimizer and isinstance(ckpt, dict) and 'optimizer' in ckpt:
        ckpt.pop('optimizer', None)
        print('Removed optimizer state to avoid param group mismatch')

    torch.save(ckpt, dst_path)
    print(f'Saved stripped checkpoint to: {dst_path}')


def main():
    parser = argparse.ArgumentParser(description='Strip occ params and/or optimizer from checkpoint')
    parser.add_argument('src', help='source checkpoint path (.pth)')
    parser.add_argument('dst', help='destination checkpoint path (.pth)')
    parser.add_argument('--keep-occ', action='store_true', help='keep occ params (default: drop)')
    parser.add_argument('--keep-optimizer', action='store_true', help='keep optimizer state (default: drop)')
    args = parser.parse_args()

    strip_checkpoint(
        args.src,
        args.dst,
        drop_occ=not args.keep_occ,
        drop_optimizer=not args.keep_optimizer,
    )


if __name__ == '__main__':
    main()
