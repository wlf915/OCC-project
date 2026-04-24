import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16


def _kl_bernoulli_from_logits(p_logits, q_logits, reduction='mean', eps=1e-6):
    """KL(p||q) for independent Bernoulli variables given logits."""
    p = torch.sigmoid(p_logits)
    q = torch.sigmoid(q_logits)
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    kl = p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
    if reduction == 'mean':
        return kl.mean()
    if reduction == 'sum':
        return kl.sum()
    if reduction == 'batchmean':
        batch = kl.shape[0] if kl.ndim > 0 else 1
        return kl.sum() / max(batch, 1)
    return kl


def _kl_categorical_from_logits(p_logits, q_logits, dim=-1, reduction='mean'):
    """KL(p||q) for categorical distributions given logits."""
    p_log_prob = F.log_softmax(p_logits, dim=dim)
    q_log_prob = F.log_softmax(q_logits, dim=dim)
    p_prob = p_log_prob.exp()
    kl = (p_prob * (p_log_prob - q_log_prob)).sum(dim=dim)
    if reduction == 'mean':
        return kl.mean()
    if reduction == 'sum':
        return kl.sum()
    if reduction == 'batchmean':
        batch = kl.shape[0] if kl.ndim > 0 else 1
        return kl.sum() / max(batch, 1)
    return kl


def _js_divergence_from_logits(p_logits, q_logits, kind='bernoulli', dim=-1, reduction='mean', eps=1e-6):
    """Jensen-Shannon divergence derived from logits."""
    if kind == 'categorical':
        p_log_prob = F.log_softmax(p_logits, dim=dim)
        q_log_prob = F.log_softmax(q_logits, dim=dim)
        p_prob = p_log_prob.exp()
        q_prob = q_log_prob.exp()
        m_prob = 0.5 * (p_prob + q_prob)
        m_prob = torch.clamp(m_prob, eps, 1.0)
        log_m = torch.log(m_prob)
        kl_pm = (p_prob * (p_log_prob - log_m)).sum(dim=dim)
        kl_qm = (q_prob * (q_log_prob - log_m)).sum(dim=dim)
    else:
        p = torch.sigmoid(p_logits)
        q = torch.sigmoid(q_logits)
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)
        m = 0.5 * (p + q)
        m = torch.clamp(m, eps, 1 - eps)
        kl_pm = p * (torch.log(p) - torch.log(m)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - m))
        kl_qm = q * (torch.log(q) - torch.log(m)) + (1 - q) * (torch.log(1 - q) - torch.log(1 - m))
    js = 0.5 * (kl_pm + kl_qm)
    if reduction == 'mean':
        return js.mean()
    if reduction == 'sum':
        return js.sum()
    if reduction == 'batchmean':
        batch = js.shape[0] if js.ndim > 0 else 1
        return js.sum() / max(batch, 1)
    return js


@HEADS.register_module()
class BEVFormerHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 occ_head=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        # occupancy head config (optional)
        self.occ_cfg = occ_head if isinstance(occ_head, dict) else None
        self.with_occ = bool(self.occ_cfg and self.occ_cfg.get('ENABLED', False))
        self.occ_loss_weight = float(self.occ_cfg.get('LOSS_WEIGHT', 0.2)) if self.with_occ else 0.0
        self.occ_per_class = bool(self.occ_cfg.get('PER_CLASS', False)) if self.with_occ else False
        self.occ_kl_enabled = bool(self.occ_cfg.get('KL_WITH_PRED', False)) if self.with_occ else False
        self.occ_kl_weight = float(self.occ_cfg.get('KL_WEIGHT', 0.1)) if self.occ_kl_enabled else 0.0
        self.occ_kl_kind = self.occ_cfg.get('KL_KIND', 'bernoulli') if self.occ_kl_enabled else 'bernoulli'
        self.occ_kl_mode = self.occ_cfg.get('KL_MODE', 'kl') if self.occ_kl_enabled else 'kl'
        self.occ_kl_reduction = self.occ_cfg.get('KL_REDUCTION', 'mean') if self.occ_kl_enabled else 'mean'
        self.occ_kl_occ_reduce = self.occ_cfg.get('KL_OCC_REDUCE', 'avg_bev') if self.occ_kl_enabled else 'avg_bev'
        self.occ_kl_pred_reduce = self.occ_cfg.get('KL_PRED_REDUCE', 'avg_query') if self.occ_kl_enabled else 'avg_query'
        self.occ_kl_detach_occ = bool(self.occ_cfg.get('KL_DETACH_OCC', False)) if self.occ_kl_enabled else False
        self.occ_kl_detach_pred = bool(self.occ_cfg.get('KL_DETACH_PRED', False)) if self.occ_kl_enabled else False
        # restrict occupancy supervision to selected classes if provided
        target_cls_cfg = self.occ_cfg.get('TARGET_CLASSES', None) if self.with_occ else None
        if target_cls_cfg is None and self.with_occ:
            target_cls_cfg = self.occ_cfg.get('TARGET_CLASS_IDS', None)
        if target_cls_cfg is None:
            self.occ_target_classes = None
        elif isinstance(target_cls_cfg, (list, tuple)):
            self.occ_target_classes = tuple(int(c) for c in target_cls_cfg)
        else:
            self.occ_target_classes = (int(target_cls_cfg),)

        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        # build occupancy branch if enabled
        if self.with_occ:
            num_convs = int(self.occ_cfg.get('NUM_CONV', 2))
            init_bias = float(self.occ_cfg.get('INIT_BIAS', 0.0))
            layers = []
            in_ch = self.embed_dims
            hid_ch = self.embed_dims
            for _ in range(max(0, num_convs - 1)):
                layers.append(nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_ch = hid_ch
            # output channels: 1 for binary, or num_classes for per-class
            out_ch = getattr(self, 'num_classes', 1) if self.occ_per_class else 1
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            self.occ_head = nn.Sequential(*layers)
            # init last bias
            with torch.no_grad():
                self.occ_head[-1].bias.fill_(init_bias)

    def _reduce_occ_logits_for_kl(self, occ_logits):
        if not self.occ_kl_enabled:
            return None
        reduce = self.occ_kl_occ_reduce
        if reduce == 'avg_bev':
            agg = occ_logits.mean(dim=(2, 3))
        elif reduce == 'max_bev':
            agg = occ_logits.amax(dim=(2, 3))
        elif reduce == 'sum_bev':
            agg = occ_logits.sum(dim=(2, 3))
        elif reduce == 'logsumexp_bev':
            agg = torch.logsumexp(occ_logits, dim=(2, 3))
            area = occ_logits.shape[2] * occ_logits.shape[3]
            if area > 0:
                agg = agg - math.log(float(area))
        else:
            raise ValueError(f'Unsupported KL_OCC_REDUCE: {reduce}')
        return agg

    def _reduce_pred_logits_for_kl(self, cls_logits):
        if not self.occ_kl_enabled:
            return None
        reduce = self.occ_kl_pred_reduce
        if reduce == 'avg_query':
            agg = cls_logits.mean(dim=1)
        elif reduce == 'max_query':
            agg = cls_logits.amax(dim=1)
        elif reduce == 'sum_query':
            agg = cls_logits.sum(dim=1)
        elif reduce == 'logsumexp_query':
            agg = torch.logsumexp(cls_logits, dim=1)
            count = cls_logits.shape[1]
            if count > 0:
                agg = agg - math.log(float(count))
        else:
            raise ValueError(f'Unsupported KL_PRED_REDUCE: {reduce}')
        return agg

    def _compute_occ_pred_kl(self, occ_logits, preds_dicts):
        if not self.occ_kl_enabled:
            return None
        if 'all_cls_scores' not in preds_dicts or preds_dicts['all_cls_scores'] is None:
            return None
        cls_logits_last = preds_dicts['all_cls_scores'][-1]
        if cls_logits_last is None:
            return None
        occ_vec_logits = self._reduce_occ_logits_for_kl(occ_logits)
        pred_vec_logits = self._reduce_pred_logits_for_kl(cls_logits_last)
        if occ_vec_logits is None or pred_vec_logits is None:
            return None

        # align shapes (B, C) or (B, 1)
        if not self.occ_per_class:
            if occ_vec_logits.dim() > 1:
                occ_vec_logits = occ_vec_logits.mean(dim=-1, keepdim=True)
            if pred_vec_logits.dim() > 1:
                pred_vec_logits = torch.logsumexp(pred_vec_logits, dim=-1, keepdim=True)
        else:
            # truncate/expand pred logits if mismatch (e.g., extra background)
            if pred_vec_logits.shape[-1] != occ_vec_logits.shape[-1]:
                min_c = min(pred_vec_logits.shape[-1], occ_vec_logits.shape[-1])
                pred_vec_logits = pred_vec_logits[..., :min_c]
                occ_vec_logits = occ_vec_logits[..., :min_c]

        if self.occ_kl_detach_occ:
            occ_vec_logits = occ_vec_logits.detach()
        if self.occ_kl_detach_pred:
            pred_vec_logits = pred_vec_logits.detach()

        reduction = self.occ_kl_reduction
        if self.occ_kl_mode == 'kl':
            if self.occ_kl_kind == 'categorical':
                loss_val = _kl_categorical_from_logits(occ_vec_logits, pred_vec_logits, dim=-1, reduction=reduction)
            else:
                loss_val = _kl_bernoulli_from_logits(occ_vec_logits, pred_vec_logits, reduction=reduction)
        elif self.occ_kl_mode == 'reverse_kl':
            if self.occ_kl_kind == 'categorical':
                loss_val = _kl_categorical_from_logits(pred_vec_logits, occ_vec_logits, dim=-1, reduction=reduction)
            else:
                loss_val = _kl_bernoulli_from_logits(pred_vec_logits, occ_vec_logits, reduction=reduction)
        elif self.occ_kl_mode == 'symmetric_kl':
            if self.occ_kl_kind == 'categorical':
                loss_forward = _kl_categorical_from_logits(occ_vec_logits, pred_vec_logits, dim=-1, reduction=reduction)
                loss_backward = _kl_categorical_from_logits(pred_vec_logits, occ_vec_logits, dim=-1, reduction=reduction)
            else:
                loss_forward = _kl_bernoulli_from_logits(occ_vec_logits, pred_vec_logits, reduction=reduction)
                loss_backward = _kl_bernoulli_from_logits(pred_vec_logits, occ_vec_logits, reduction=reduction)
            loss_val = 0.5 * (loss_forward + loss_backward)
        elif self.occ_kl_mode == 'js':
            if self.occ_kl_kind == 'categorical':
                loss_val = _js_divergence_from_logits(occ_vec_logits, pred_vec_logits, kind='categorical', dim=-1, reduction=reduction)
            else:
                loss_val = _js_divergence_from_logits(occ_vec_logits, pred_vec_logits, kind='bernoulli', reduction=reduction)
        else:
            raise ValueError(f'Unsupported KL_MODE: {self.occ_kl_mode}')

        return torch.nan_to_num(loss_val)

    def _build_occ_class_mask(self, labels):
        """Create boolean mask for classes supervised by occupancy."""
        if labels is None:
            return None
        if self.occ_target_classes is None:
            return torch.ones_like(labels, dtype=torch.bool)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for cls_id in self.occ_target_classes:
            if cls_id < 0:
                continue
            mask |= (labels == cls_id)
        return mask

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

            bev_embed, hs, init_reference, inter_references = outputs
            hs = hs.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                # TODO: check the shape of reference
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                 self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                 self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                 self.pc_range[2]) + self.pc_range[2])

                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
            }

            # Optional occupancy logits from BEV feature
            if self.with_occ:
                B = mlvl_feats[0].shape[0]
                # bev_embed returned from transformer has shape (H*W, B, C)
                assert bev_embed.shape[0] == self.bev_h * self.bev_w, 'Unexpected BEV shape'
                bev_feat = (
                    bev_embed.permute(1, 0, 2)  # (B, H*W, C)
                    .contiguous()
                    .view(B, self.bev_h, self.bev_w, self.embed_dims)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                occ_logits = self.occ_head(bev_feat)  # (B, {1|C}, H, W)
                outs['occ_logits'] = occ_logits

            return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # Occupancy auxiliary loss
        if self.with_occ and 'occ_logits' in preds_dicts:
            occ_logits = preds_dicts['occ_logits']  # (B,{1|C},H,W)
            if self.occ_per_class:
                with torch.no_grad():
                    occ_targets, occ_weights = self._build_occ_targets_per_class(gt_bboxes_list, gt_labels_list, device=occ_logits.device)  # (B,C,H,W)
                # compute pos/neg using weights (ignored cells have weight 0)
                B, C, H, W = occ_targets.shape
                weight_sum = occ_weights.sum(dim=(0, 2, 3)).clamp(min=1e-6)  # (C,)
                pos = (occ_targets.float() * occ_weights).sum(dim=(0, 2, 3))  # (C,)
                neg = weight_sum - pos
                pos_weight = torch.where(pos > 0, neg / pos.clamp(min=1.0), torch.ones_like(pos))
                pw = pos_weight.view(1, C, 1, 1)
                bce = F.binary_cross_entropy_with_logits(
                    occ_logits,
                    occ_targets.float(),
                    pos_weight=pw,
                    weight=occ_weights,
                    reduction='sum')
                norm = occ_weights.sum().clamp(min=1.0)
                occ_loss = bce / norm
                loss_dict['loss_occ'] = occ_loss * self.occ_loss_weight
                # optional: log per-class positive ratios among unmasked cells
                for ci in range(min(3, occ_targets.shape[1])):  # log first few to avoid spam
                    loss_dict[f'occ_pos_c{ci}'] = pos[ci] / weight_sum[ci]
            else:
                with torch.no_grad():
                    occ_targets, occ_weights = self._build_occ_targets(gt_bboxes_list, gt_labels_list, device=occ_logits.device)  # (B,H,W)
                pos = (occ_targets.float() * occ_weights).sum()
                weight_sum = occ_weights.sum().clamp(min=1.0)
                neg = weight_sum - pos
                pos_weight = 1.0 if pos.item() == 0 else float(neg / pos.clamp(min=1.0))
                pw = torch.tensor(pos_weight, device=occ_logits.device)
                bce = F.binary_cross_entropy_with_logits(
                    occ_logits.squeeze(1),
                    occ_targets.float(),
                    pos_weight=pw,
                    weight=occ_weights,
                    reduction='sum')
                occ_loss = bce / weight_sum
                loss_dict['loss_occ'] = occ_loss * self.occ_loss_weight

            if self.occ_kl_enabled:
                kl_val = self._compute_occ_pred_kl(occ_logits, preds_dicts)
                if kl_val is not None:
                    loss_dict['loss_occ_pred_kl'] = kl_val * self.occ_kl_weight

        return loss_dict

    def _build_occ_targets(self, gt_bboxes_list, gt_labels_list=None, device=None):
        """Build BEV occupancy targets and weights.

        - targets: (B, H, W) bool, 1 if a target-class box occupies the cell.
        - weights: (B, H, W) float, 0 means ignore in loss. Cells covered *only* by
            non-target classes are ignored (weight=0) instead of treated as negatives.
        """
        bs = len(gt_bboxes_list)
        H, W = self.bev_h, self.bev_w
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H
        x_centers = torch.linspace(x_min + dx/2, x_max - dx/2, W, device=device)
        y_centers = torch.linspace(y_min + dy/2, y_max - dy/2, H, device=device)
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers)  # (H,W)
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (H*W,2)
        targets, weights = [], []
        for i in range(bs):
            item = gt_bboxes_list[i]
            labels = gt_labels_list[i] if gt_labels_list is not None else None
            if labels is not None:
                labels = labels.to(device)
            if hasattr(item, 'tensor') and hasattr(item, 'gravity_center'):
                gtb = torch.cat((item.gravity_center, item.tensor[:, 3:]), dim=1).to(device)
            elif torch.is_tensor(item):
                gtb = item.to(device)
            else:
                gtb = torch.empty((0, 7), device=device)

            occ_target = torch.zeros(pts.shape[0], device=device, dtype=torch.bool)
            occ_excluded = torch.zeros_like(occ_target)

            if gtb.numel() > 0:
                # split into target-class boxes and excluded boxes
                class_mask = self._build_occ_class_mask(labels)
                if class_mask is not None:
                    # target boxes
                    if class_mask.any():
                        gtb_t = gtb[class_mask]
                    else:
                        gtb_t = torch.empty((0, 7), device=device)
                    # non-target boxes
                    gtb_nt = gtb[~class_mask] if class_mask.numel() else torch.empty((0, 7), device=device)
                else:
                    gtb_t = gtb
                    gtb_nt = torch.empty((0, 7), device=device)

                def _mark_occ(boxes):
                    if boxes.numel() == 0:
                        return torch.zeros_like(occ_target)
                    cx = boxes[:, 0:1]
                    cy = boxes[:, 1:2]
                    dx_box = boxes[:, 3:4]
                    dy_box = boxes[:, 4:5]
                    yaw = boxes[:, 6:7]
                    cos_y = torch.cos(yaw)
                    sin_y = torch.sin(yaw)
                    p = pts[None, :, :]  # (1,P,2)
                    pcx = p[..., 0] - cx  # (N,P)
                    pcy = p[..., 1] - cy
                    x_loc = cos_y * pcx + sin_y * pcy
                    y_loc = -sin_y * pcx + cos_y * pcy
                    in_box = (x_loc.abs() <= dx_box/2) & (y_loc.abs() <= dy_box/2)  # (N,P)
                    return in_box.any(dim=0)

                occ_target = _mark_occ(gtb_t)
                occ_excluded = _mark_occ(gtb_nt)

            # weight=0 only for cells covered by non-target boxes and not by target boxes
            w = torch.ones_like(occ_target, dtype=torch.float)
            w[(occ_excluded & (~occ_target))] = 0.0
            targets.append(occ_target)
            weights.append(w)

        targets = torch.stack(targets, dim=0).view(bs, H, W)
        weights = torch.stack(weights, dim=0).view(bs, H, W)
        return targets, weights

    def _build_occ_targets_per_class(self, gt_bboxes_list, gt_labels_list, device):
        """Build per-class BEV occupancy targets and weights.

        Returns:
            targets: (B, C, H, W) bool, per-class occupancy for target classes.
            weights: (B, C, H, W) float, 0 to ignore cells occupied only by non-target classes.
        """
        bs = len(gt_bboxes_list)
        H, W = self.bev_h, self.bev_w
        C = getattr(self, 'num_classes', 1)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H
        x_centers = torch.linspace(x_min + dx/2, x_max - dx/2, W, device=device)
        y_centers = torch.linspace(y_min + dy/2, y_max - dy/2, H, device=device)
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers)  # (H,W)
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (H*W,2)
        batch_targets, batch_weights = [] , []
        for i in range(bs):
            item = gt_bboxes_list[i]
            labels = gt_labels_list[i].to(device)
            if hasattr(item, 'tensor') and hasattr(item, 'gravity_center'):
                gtb = torch.cat((item.gravity_center, item.tensor[:, 3:]), dim=1).to(device)
            elif torch.is_tensor(item):
                gtb = item.to(device)
            else:
                gtb = torch.empty((0, 7), device=device)

            # prepare splits
            class_mask = self._build_occ_class_mask(labels)
            if class_mask is not None:
                gtb_t = gtb[class_mask] if class_mask.any() else torch.empty((0, 7), device=device)
                labels_t = labels[class_mask] if class_mask.any() else torch.empty((0,), device=device, dtype=labels.dtype)
                gtb_nt = gtb[~class_mask] if class_mask.numel() else torch.empty((0, 7), device=device)
            else:
                gtb_t = gtb
                labels_t = labels
                gtb_nt = torch.empty((0, 7), device=device)

            def _mark_occ(boxes):
                if boxes.numel() == 0:
                    return torch.zeros(C, pts.shape[0], device=device, dtype=torch.bool)
                cx = boxes[:, 0:1]
                cy = boxes[:, 1:2]
                dx_box = boxes[:, 3:4]
                dy_box = boxes[:, 4:5]
                yaw = boxes[:, 6:7]
                cos_y = torch.cos(yaw)
                sin_y = torch.sin(yaw)
                p = pts[None, :, :]  # (1,P,2)
                pcx = p[..., 0] - cx  # (N,P)
                pcy = p[..., 1] - cy
                x_loc = cos_y * pcx + sin_y * pcy
                y_loc = -sin_y * pcx + cos_y * pcy
                in_box = (x_loc.abs() <= dx_box/2) & (y_loc.abs() <= dy_box/2)  # (N,P)
                occ = torch.zeros(C, in_box.shape[1], device=device, dtype=torch.bool)
                target_classes = self.occ_target_classes if self.occ_target_classes is not None else range(C)
                for c in target_classes:
                    if c < 0 or c >= C or labels_t.numel() == 0:
                        continue
                    mask = (labels_t == c)
                    if mask.any():
                        occ[c] = in_box[mask].any(dim=0)
                return occ

            cls_occ = _mark_occ(gtb_t)

            # excluded occupancy (non-target boxes) -> used to mask out
            def _mark_any(boxes):
                if boxes.numel() == 0:
                    return torch.zeros(pts.shape[0], device=device, dtype=torch.bool)
                cx = boxes[:, 0:1]
                cy = boxes[:, 1:2]
                dx_box = boxes[:, 3:4]
                dy_box = boxes[:, 4:5]
                yaw = boxes[:, 6:7]
                cos_y = torch.cos(yaw)
                sin_y = torch.sin(yaw)
                p = pts[None, :, :]
                pcx = p[..., 0] - cx
                pcy = p[..., 1] - cy
                x_loc = cos_y * pcx + sin_y * pcy
                y_loc = -sin_y * pcx + cos_y * pcy
                in_box = (x_loc.abs() <= dx_box/2) & (y_loc.abs() <= dy_box/2)
                return in_box.any(dim=0)

            occ_excluded = _mark_any(gtb_nt)
            occ_target_any = cls_occ.any(dim=0)

            w = torch.ones_like(cls_occ, dtype=torch.float)
            ignore_mask = occ_excluded & (~occ_target_any)
            w[:, ignore_mask] = 0.0

            batch_targets.append(cls_occ)
            batch_weights.append(w)

        targets = torch.stack(batch_targets, dim=0).view(bs, C, H, W)
        weights = torch.stack(batch_weights, dim=0).view(bs, C, H, W)
        return targets, weights

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list


@HEADS.register_module()
class BEVFormerHead_GroupDETR(BEVFormerHead):
    def __init__(self,
                 *args,
                 group_detr=1,
                 **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:  # NOTE: Only difference to bevformer head
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        assert enc_cls_scores is None and enc_bbox_preds is None 

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        loss_dict = dict()
        loss_dict['loss_cls'] = 0
        loss_dict['loss_bbox'] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = 0
        num_query_per_group = self.num_query // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index+1) * num_query_per_group
            group_cls_scores =  all_cls_scores[:, :,group_query_start:group_query_end, :]
            group_bbox_preds = all_bbox_preds[:, :,group_query_start:group_query_end, :]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, group_cls_scores, group_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list)
            loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
            loss_dict['loss_bbox'] += losses_bbox[-1] / self.group_detr
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / self.group_detr
                loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i / self.group_detr
                num_dec_layer += 1
        return loss_dict