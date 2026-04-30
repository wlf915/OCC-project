# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import os
import json
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from mmcv import mkdir_or_exist
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.analysis_export_cfg = self._build_analysis_export_cfg()

    def _build_analysis_export_cfg(self):
        enabled = os.getenv('BEVFORMER_EXPORT_ANALYSIS', '0') == '1'
        out_dir = os.getenv('BEVFORMER_EXPORT_DIR', os.path.join('artifacts', 'analysis_export'))
        save_occ = os.getenv('BEVFORMER_EXPORT_OCC', '1') == '1'
        save_bev = os.getenv('BEVFORMER_EXPORT_BEV', '1') == '1'
        save_cls = os.getenv('BEVFORMER_EXPORT_CLS', '1') == '1'
        save_pred = os.getenv('BEVFORMER_EXPORT_PRED', '1') == '1'
        dtype = os.getenv('BEVFORMER_EXPORT_DTYPE', 'float16').lower()

        cfg = getattr(self, 'test_cfg', None)
        if isinstance(cfg, dict):
            export_cfg = cfg.get('analysis_export', None)
            if isinstance(export_cfg, dict):
                enabled = bool(export_cfg.get('enabled', enabled))
                out_dir = export_cfg.get('out_dir', out_dir)
                save_occ = bool(export_cfg.get('save_occ', save_occ))
                save_bev = bool(export_cfg.get('save_bev', save_bev))
                save_cls = bool(export_cfg.get('save_cls', save_cls))
                save_pred = bool(export_cfg.get('save_pred', save_pred))
                dtype = str(export_cfg.get('dtype', dtype)).lower()

        if dtype not in ('float16', 'float32'):
            dtype = 'float16'

        return dict(
            enabled=enabled,
            out_dir=out_dir,
            save_occ=save_occ,
            save_bev=save_bev,
            save_cls=save_cls,
            save_pred=save_pred,
            dtype=dtype,
        )

    def _get_rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _safe_sample_id(self, meta, fallback_idx):
        sid = None
        if isinstance(meta, dict):
            for key in ('sample_token', 'sample_idx', 'scene_token', 'frame_id'):
                if key in meta:
                    sid = str(meta[key])
                    break
        if sid is None:
            sid = str(fallback_idx)
        sid = sid.replace('/', '_')
        return sid

    def _append_jsonl(self, path, obj):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    def _save_analysis_artifacts(self, outs, bbox_list, img_metas):
        cfg = self.analysis_export_cfg
        if not cfg.get('enabled', False):
            return

        try:
            rank = self._get_rank()
            root_dir = cfg['out_dir']
            occ_dir = os.path.join(root_dir, 'occ')
            bev_dir = os.path.join(root_dir, 'bev')
            cls_dir = os.path.join(root_dir, 'cls')
            pred_dir = os.path.join(root_dir, 'pred')
            meta_dir = os.path.join(root_dir, 'meta')
            mkdir_or_exist(root_dir)
            mkdir_or_exist(meta_dir)
            if cfg.get('save_occ', True):
                mkdir_or_exist(occ_dir)
            if cfg.get('save_bev', True):
                mkdir_or_exist(bev_dir)
            if cfg.get('save_cls', True):
                mkdir_or_exist(cls_dir)
            if cfg.get('save_pred', True):
                mkdir_or_exist(pred_dir)

            np_dtype = np.float16 if cfg.get('dtype') == 'float16' else np.float32

            occ_probs = None
            occ_logits = outs.get('occ_logits', None)
            if occ_logits is not None:
                occ_probs = torch.sigmoid(occ_logits).detach().cpu().numpy().astype(np_dtype)

            bev_embed = outs.get('bev_embed', None)
            bev_feat = None
            if bev_embed is not None:
                bev_embed = bev_embed.detach().float()
                if bev_embed.dim() == 3:
                    if bev_embed.shape[0] == self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w:
                        bev_feat = (
                            bev_embed.permute(1, 0, 2)
                            .contiguous()
                            .view(
                                bev_embed.shape[1],
                                self.pts_bbox_head.bev_h,
                                self.pts_bbox_head.bev_w,
                                self.pts_bbox_head.embed_dims,
                            )
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                    elif bev_embed.shape[1] == self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w:
                        bev_feat = (
                            bev_embed.contiguous()
                            .view(
                                bev_embed.shape[0],
                                self.pts_bbox_head.bev_h,
                                self.pts_bbox_head.bev_w,
                                bev_embed.shape[2],
                            )
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
            bev_np = None if bev_feat is None else bev_feat.detach().cpu().numpy().astype(np_dtype)

            cls_last = None
            all_cls_scores = outs.get('all_cls_scores', None)
            if all_cls_scores is not None:
                cls_last = all_cls_scores[-1].detach().cpu().numpy().astype(np_dtype)

            meta_path = os.path.join(meta_dir, f'meta_rank{rank}.jsonl')

            for i, meta in enumerate(img_metas):
                sid = self._safe_sample_id(meta, i)
                record = {
                    'sample_id': sid,
                    'rank': rank,
                    'index_in_batch': i,
                    'paths': {},
                }
                if isinstance(meta, dict):
                    record['meta'] = {
                        'sample_token': meta.get('sample_token', None),
                        'sample_idx': meta.get('sample_idx', None),
                        'scene_token': meta.get('scene_token', None),
                        'frame_id': meta.get('frame_id', None),
                        'timestamp': meta.get('timestamp', None),
                    }

                if cfg.get('save_occ', True) and occ_probs is not None and i < occ_probs.shape[0]:
                    occ_path = os.path.join(occ_dir, f'{sid}_occ.npy')
                    np.save(occ_path, occ_probs[i])
                    record['paths']['occ'] = occ_path

                if cfg.get('save_bev', True) and bev_np is not None and i < bev_np.shape[0]:
                    bev_path = os.path.join(bev_dir, f'{sid}_bev.npy')
                    np.save(bev_path, bev_np[i])
                    record['paths']['bev'] = bev_path

                if cfg.get('save_cls', True) and cls_last is not None and i < cls_last.shape[0]:
                    cls_path = os.path.join(cls_dir, f'{sid}_cls_last.npy')
                    np.save(cls_path, cls_last[i])
                    record['paths']['cls_last'] = cls_path

                if cfg.get('save_pred', True) and i < len(bbox_list):
                    bboxes, scores, labels = bbox_list[i]
                    box_path = os.path.join(pred_dir, f'{sid}_boxes.npy')
                    score_path = os.path.join(pred_dir, f'{sid}_scores.npy')
                    label_path = os.path.join(pred_dir, f'{sid}_labels.npy')
                    np.save(box_path, bboxes.tensor.detach().cpu().numpy().astype(np_dtype))
                    np.save(score_path, scores.detach().cpu().numpy().astype(np_dtype))
                    np.save(label_path, labels.detach().cpu().numpy().astype(np.int64))
                    record['paths']['boxes'] = box_path
                    record['paths']['scores'] = score_path
                    record['paths']['labels'] = label_path

                self._append_jsonl(meta_path, record)
        except Exception:
            pass


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        # If occupancy logits are present, convert to probabilities and save
        # per-sample under `work_dirs/occ_maps/` as .npy.
        # The heatmap visualization is intentionally kept out of inference and
        # can be generated later from the saved arrays.
        try:
            occ_logits = outs.get('occ_logits', None)
            if occ_logits is not None:
                out_dir = os.path.join('work_dirs', 'occ_maps')
                mkdir_or_exist(out_dir)

                # occ_logits shape: (B, C|1, H, W)
                occ_probs = torch.sigmoid(occ_logits).detach().cpu().numpy()
                B = occ_probs.shape[0]
                for i in range(B):
                    meta = img_metas[i] if i < len(img_metas) else {}
                    # try sample identifier keys then fallback to index
                    sid = None
                    for key in ('sample_token', 'sample_idx', 'scene_token', 'frame_id'):
                        if isinstance(meta, dict) and key in meta:
                            sid = meta[key]
                            break
                    if sid is None:
                        sid = str(i)

                    npy_path = os.path.join(out_dir, f"{sid}_occ.npy")
                    np.save(npy_path, occ_probs[i])
        except Exception:
            # don't break inference on save errors
            pass

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        self._save_analysis_artifacts(outs, bbox_list, img_metas)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
