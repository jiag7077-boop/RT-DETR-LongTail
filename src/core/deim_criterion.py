"""
[MODIFIED] DEIM criterion with SPFM (Self-Paced Focused Matching).

All SPFM additions are marked with ``# [NEW: SPFM]`` and correspond to
Section 3.5 of the paper. Two components are integrated into the existing
MAL (Matching-Aware Loss):

  1. **Bilateral Gaussian modulation** (w_focus):
     Quality-aware sample weighting based on IoU between matched
     predictions and targets. Uses robust centre estimation via
     median → EMA → interval clamping.

  2. **Class-level priority weights** (w_spl):
     Temperature-scheduled softmax over per-class easiness scores
     from teacher evaluation, blended with uniform weights.

Both components are guarded by config flags and can be independently
disabled for ablation studies.

Based on DEIM (https://github.com/ShihuaHuang95/DEIM).
Modified from D-FINE (https://github.com/Peterande/D-FINE/).
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision
import copy
import logging
from pathlib import Path

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized, is_main_process
from ..core import register

# [NEW: SPFM] Per-class statistics for priority weight computation
try:
    from ..dataset_info import TOTAL_CLASSES, NUM_SAMPLES_PER_CLASS
except ImportError:
    raise ImportError(
        "Cannot import from engine.dataset_info. "
        "Please ensure dataset_info.py is properly configured."
    )

logger = logging.getLogger(__name__)


@register()
class DEIMCriterion(nn.Module):
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0,
                 num_classes=80, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, mal_alpha=None, use_uni_set=True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

        # ── [NEW: SPFM] Robust focus centre via EMA (Section 3.5) ──
        self.adaptive_momentum = 0.99           # EMA momentum m
        self.register_buffer('adaptive_mu', torch.tensor(0.45))  # μ_focus

        self.output_dir = None
        self._spfm_step_count = 0

        # ── Original DEIM fields ──
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

        # ── [NEW: SPFM] Configuration defaults ──
        self.use_spfm_priority = False          # class-level priority weights
        self.use_spfm_focus = False              # bilateral Gaussian modulation
        self.total_epochs = -1
        self.tau_start = 0.1                     # priority temperature start
        self.tau_end = 1.0                       # priority temperature end
        self.lambda_smooth = 0.1                 # blend factor with uniform
        self.p_mal_gamma = 2.0                   # MAL gamma exponent

        # Per-class priority weight vector (updated by solver each epoch)
        self.w_priority_vector = torch.ones(TOTAL_CLASSES)
        self.current_epoch = -1

    # ================================================================
    # [NEW: SPFM] Config injection from solver
    # ================================================================
    def set_cfg_externally(self, cfg):
        """
        Inject training config from the solver. Reads SPFM parameters
        from the YAML config and stores them as instance attributes.

        Called once at the start of training by det_solver.py.
        """
        logger.info("Config injected into DEIMCriterion from solver.")
        self.cfg = cfg

        # SPFM priority weights (class-level, Section 3.5)
        spfm_cfg = cfg.yaml_cfg.get('SPFM', {})
        self.use_spfm_priority = spfm_cfg.get('USE', False)
        self.total_epochs = spfm_cfg.get('TOTAL_EPOCHS', 120)
        self.tau_start = spfm_cfg.get('TAU_START', 0.1)
        self.tau_end = spfm_cfg.get('TAU_END', 1.0)
        self.lambda_smooth = spfm_cfg.get('LAMBDA_SMOOTH', 0.1)

        # SPFM bilateral Gaussian focus (sample-level, Section 3.5)
        # Controlled by a sub-key; can be merged into SPFM config
        self.use_spfm_focus = spfm_cfg.get('USE_FOCUS', True)
        self.p_mal_gamma = spfm_cfg.get('GAMMA', 2.0)

    # ================================================================
    # [NEW: SPFM] Priority weight update (called by solver each epoch)
    # ================================================================
    def update_priority_weights(self, E_old_buffer, current_epoch):
        """
        Update class-level priority weights using teacher easiness scores.

        Implements temperature-scheduled softmax over easiness scores:
            w_c = C * softmax(E_c / τ_t)
        where τ_t linearly increases from τ_start to τ_end, and the
        result is blended with uniform weights via λ_smooth.

        Early in training (small τ), the softmax is sharp and assigns
        much higher weight to difficult (low-easiness) classes. As
        training progresses (large τ), the distribution flattens toward
        uniform, preventing over-correction.

        Args:
            E_old_buffer: list[float] of length TOTAL_CLASSES,
                          per-class easiness scores from teacher evaluation
            current_epoch: current training epoch
        """
        if not self.use_spfm_priority:
            self.w_priority_vector.fill_(1.0)
            return

        self.current_epoch = current_epoch

        # Temperature schedule: τ increases linearly with training progress
        progress = (current_epoch + 1) / self.total_epochs
        tau_t = self.tau_start + (self.tau_end - self.tau_start) * progress

        # Temperature-scaled softmax over easiness scores
        E_stable = torch.tensor(
            [float(e) for e in E_old_buffer],
            device=self.w_priority_vector.device,
        )
        w_priority_raw = TOTAL_CLASSES * torch.softmax(E_stable / tau_t, dim=0)

        # Smooth blend with uniform weights (Eq. in Section 3.5)
        # λ=0 → fully adaptive, λ=1 → fully uniform
        w_priority = ((1.0 - self.lambda_smooth) * w_priority_raw
                      + self.lambda_smooth * 1.0)
        self.w_priority_vector = w_priority

        if is_main_process():
            logger.info(
                f"[SPFM] Epoch {current_epoch} priority weights "
                f"(first 5): "
                f"{[round(w.item(), 3) for w in self.w_priority_vector[:5]]}"
            )

    # ================================================================
    # Standard DEIM losses (unchanged)
    # ================================================================
    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """Sigmoid focal loss for classification (original DEIM)."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes,
                        values=None):
        """Varifocal loss for classification (original DEIM)."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(
            target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = (self.alpha * pred_score.pow(self.gamma) * (1 - target)
                  + target_score)

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    # ================================================================
    # [MODIFIED] MAL loss with SPFM integration (Section 3.5)
    # ================================================================
    def loss_labels_mal(self, outputs, targets, indices, num_boxes,
                        values=None):
        """
        Matching-Aware Loss with SPFM modulation.

        This is the original DEIM MAL loss, extended with two SPFM
        components that modulate the positive-sample weights:

        1. **Bilateral Gaussian focus** (w_focus, Eq. 5-7):
           Computes a quality-aware weight for each matched positive
           sample based on its IoU. The focus centre μ is robustly
           estimated via median → EMA → interval clamping.

        2. **Class-level priority** (w_spl):
           Per-class weights from temperature-scheduled teacher
           evaluation, encouraging the model to focus on difficult
           (typically tail) classes.

        The combined weight ``w_spl * w_focus`` is normalised to have
        unit mean over positive positions, then clamped to [0.1, 3.0]
        for training stability.

        When both SPFM components are disabled, this reduces exactly
        to the original DEIM MAL loss.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # ── Compute matched IoUs ──
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        # ── Standard MAL setup ──
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(
            target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)

        # Negative-sample weight (standard MAL)
        if self.mal_alpha is not None:
            base_weight_neg = (self.mal_alpha
                               * pred_score.pow(self.gamma)
                               * (1 - target))
        else:
            base_weight_neg = pred_score.pow(self.gamma) * (1 - target)

        # ==============================================================
        # [NEW: SPFM] Component 1 — Bilateral Gaussian focus (Eq. 5-7)
        # ==============================================================
        w_focus = torch.ones_like(
            target_score, dtype=src_logits.dtype)

        if self.use_spfm_focus:
            if ious.numel() == 0:
                # No matched positives in this batch
                w_focus_pos = torch.zeros_like(
                    target_classes, dtype=src_logits.dtype)
                w_focus = w_focus_pos.unsqueeze(-1)
            else:
                # Step 1: Robust centre estimation
                #   median → EMA → interval clamping
                if self.training:
                    batch_median = ious.detach().median()
                    if not torch.isnan(batch_median):
                        self.adaptive_mu = (
                            self.adaptive_momentum * self.adaptive_mu
                            + (1 - self.adaptive_momentum)
                            * batch_median.item()
                        )

                # Interval clamping: μ ∈ [0.35, 0.65]
                mu_val = max(0.35, min(0.65, self.adaptive_mu.item()))
                sigma = 0.22    # Fixed Gaussian bandwidth

                # Step 2: Bilateral Gaussian weight (Eq. 6)
                #   w_i = exp(-(q_i - μ)² / 2σ²)
                w_focus_per_target = torch.exp(
                    -(ious.detach() - mu_val) ** 2 / (2 * sigma ** 2)
                )

                # Step 3: Hard validity mask (Eq. 7)
                #   Suppress very low-quality matches (IoU < 0.15)
                valid_mask = (ious.detach() >= 0.15).float()
                w_focus_per_target = w_focus_per_target * valid_mask

                # Map per-target weights back to (B, Q) layout
                w_focus_pos = torch.zeros_like(
                    target_classes, dtype=src_logits.dtype)
                w_focus_pos[idx] = w_focus_per_target.to(
                    w_focus_pos.dtype)
                w_focus = w_focus_pos.unsqueeze(-1)     # (B, Q, 1)

        # ==============================================================
        # [NEW: SPFM] Component 2 — Class-level priority weights
        # ==============================================================
        if self.use_spfm_priority:
            # w_priority_vector shape: (C,) → broadcast to (1, 1, C)
            w_spl = self.w_priority_vector.to(
                src_logits.device)[None, None, :]
        else:
            w_spl = 1.0

        # ==============================================================
        # [NEW: SPFM] Combine and normalise
        # ==============================================================
        priority_weight_map = w_spl * w_focus

        if isinstance(priority_weight_map, torch.Tensor):
            with torch.no_grad():
                pos_mask = (target > 0)
                if priority_weight_map.shape != pos_mask.shape:
                    priority_weight_map = priority_weight_map.expand_as(
                        pos_mask)
                effective_pos = pos_mask & (priority_weight_map > 0)

                if effective_pos.sum() > 0:
                    # Mean-one normalisation (preserves overall loss scale)
                    mean_w = priority_weight_map[effective_pos].mean()
                    scale = torch.clamp(
                        1.0 / (mean_w + 1e-6), max=2.76)
                    priority_weight_map = torch.where(
                        effective_pos,
                        priority_weight_map * scale,
                        priority_weight_map,
                    )
                    # Stability clamping
                    priority_weight_map = torch.where(
                        effective_pos,
                        torch.clamp(priority_weight_map, min=0.1, max=3.0),
                        priority_weight_map,
                    )

        # ── [NEW: SPFM] Optional logging of adaptive μ ──
        if (self.training and self.use_spfm_focus):
            self._spfm_step_count += 1
            if (self._spfm_step_count % 100 == 0
                    and is_main_process()
                    and self.output_dir is not None):
                probe_dir = Path(self.output_dir) / "stats"
                probe_dir.mkdir(parents=True, exist_ok=True)
                log_path = probe_dir / "spfm_mu_log.csv"
                if not log_path.exists():
                    with open(log_path, 'w') as f:
                        f.write("step,mu_ema,sigma\n")
                with open(log_path, 'a') as f:
                    f.write(
                        f"{self._spfm_step_count},"
                        f"{self.adaptive_mu.item():.4f},"
                        f"{sigma}\n"
                    )

        # ── Assemble final weight and compute loss ──
        weight_pos = target * priority_weight_map
        weight = base_weight_neg + weight_pos

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    # ================================================================
    # Box regression loss (original DEIM, unchanged)
    # ================================================================
    def loss_boxes(self, outputs, targets, indices, num_boxes,
                   boxes_weight=None):
        """L1 + GIoU box regression loss (original DEIM)."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = (loss_giou if boxes_weight is None
                     else loss_giou * boxes_weight)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    # ================================================================
    # Local distribution loss (original DEIM, unchanged)
    # ================================================================
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """FGL + DDF local distribution loss (original DEIM)."""
        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            pred_corners = outputs['pred_corners'][idx].reshape(
                -1, (self.reg_max + 1))
            ref_points = outputs['ref_points'][idx].detach()

            with torch.no_grad():
                if (self.fgl_targets_dn is None
                        and 'is_dn' in outputs):
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max, outputs['reg_scale'],
                        outputs['up'])
                if (self.fgl_targets is None
                        and 'is_dn' not in outputs):
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max, outputs['reg_scale'],
                        outputs['up'])

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if 'is_dn' in outputs
                else self.fgl_targets)

            ious = torch.diag(box_iou(
                box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),
                box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(
                1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right,
                weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(
                    -1, (self.reg_max + 1))
                target_corners = outputs['teacher_corners'].reshape(
                    -1, (self.reg_max + 1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = (
                        outputs['teacher_logits'].sigmoid()
                        .max(dim=-1)[0])
                    mask = torch.zeros_like(
                        weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(
                        1, 1, 4).reshape(-1)
                    weight_targets_local[idx] = (
                        ious.reshape_as(weight_targets_local[idx])
                        .to(weight_targets_local.dtype))
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1)
                        .repeat(1, 1, 4).reshape(-1).detach())

                    loss_match_local = (
                        weight_targets_local * (T ** 2) * (
                            nn.KLDivLoss(reduction='none')(
                                F.log_softmax(
                                    pred_corners / T, dim=1),
                                F.softmax(
                                    target_corners.detach() / T,
                                    dim=1),
                            )).sum(-1))

                    if 'is_dn' not in outputs:
                        batch_scale = (
                            8 / outputs['pred_boxes'].shape[0])
                        self.num_pos = (
                            mask.sum() * batch_scale) ** 0.5
                        self.num_neg = (
                            (~mask).sum() * batch_scale) ** 0.5

                    loss_local1 = (loss_match_local[mask].mean()
                                   if mask.any() else 0)
                    loss_local2 = (loss_match_local[~mask].mean()
                                   if (~mask).any() else 0)
                    losses['loss_ddf'] = (
                        (loss_local1 * self.num_pos
                         + loss_local2 * self.num_neg)
                        / (self.num_pos + self.num_neg))

        return losses

    # ================================================================
    # Index helpers (original DEIM, unchanged)
    # ================================================================
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i)
             for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i)
             for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Unified matching indices across decoder layers (DEIM)."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]),
                 torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(
                    indices.copy(), indices_aux.copy())
            ]
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1)
                    for idx in indices]:
            unique, counts = torch.unique(
                ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(
                counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx_pair in unique_sorted:
                row_idx = idx_pair[0].item()
                col_idx = idx_pair[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(
                list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(
                list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    # ================================================================
    # Loss dispatcher
    # ================================================================
    def get_loss(self, loss, outputs, targets, indices, num_boxes,
                 **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](
            outputs, targets, indices, num_boxes, **kwargs)

    # ================================================================
    # Forward (original DEIM structure, unchanged)
    # ================================================================
    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(
            outputs_without_aux, targets)['indices']
        self._clear_cache()

        if 'aux_outputs' in outputs:
            indices_aux_list = []
            cached_indices, cached_indices_enc = [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = (
                    outputs['aux_outputs'] + [outputs['pre_outputs']])
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(
                    aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(
                    outputs['enc_aux_outputs']):
                indices_enc = self.matcher(
                    aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(
                indices, indices_aux_list)
            num_boxes_go = sum(
                len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float,
                device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(
                num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(
            num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            use_uni = self.use_uni_set and (
                loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni else indices
            num_in = num_boxes_go if use_uni else num_boxes
            meta = self.get_loss_meta_info(
                loss, outputs, targets, indices_in)
            l_dict = self.get_loss(
                loss, outputs, targets, indices_in, num_in,
                **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k]
                      for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # Auxiliary decoder layer losses
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(
                    outputs['aux_outputs']):
                if 'local' in self.losses:
                    aux_outputs['up'] = outputs['up']
                    aux_outputs['reg_scale'] = outputs['reg_scale']
                for loss in self.losses:
                    use_uni = self.use_uni_set and (
                        loss in ['boxes', 'local'])
                    indices_in = (indices_go if use_uni
                                  else cached_indices[i])
                    num_in = (num_boxes_go if use_uni
                              else num_boxes)
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in,
                        num_in, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v
                              for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Pre-outputs loss
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                use_uni = self.use_uni_set and (
                    loss in ['boxes', 'local'])
                indices_in = (indices_go if use_uni
                              else cached_indices[-1])
                num_in = (num_boxes_go if use_uni
                          else num_boxes)
                meta = self.get_loss_meta_info(
                    loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(
                    loss, aux_outputs, targets, indices_in,
                    num_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k]
                          for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v
                          for k, v in l_dict.items()}
                losses.update(l_dict)

        # Encoder auxiliary losses
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(
                    outputs['enc_aux_outputs']):
                for loss in self.losses:
                    use_uni = self.use_uni_set and (loss == 'boxes')
                    indices_in = (indices_go if use_uni
                                  else cached_indices_enc[i])
                    num_in = (num_boxes_go if use_uni
                              else num_boxes)
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, enc_targets,
                        indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets,
                        indices_in, num_in, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v
                              for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # Denoising losses
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs
            indices_dn = self.get_cdn_matched_indices(
                outputs['dn_meta'], targets)
            dn_num_boxes = (
                num_boxes * outputs['dn_meta']['dn_num_group'])

            for i, aux_outputs in enumerate(
                    outputs['dn_outputs']):
                if 'local' in self.losses:
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'] = outputs['up']
                    aux_outputs['reg_scale'] = outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn,
                        dn_num_boxes, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v
                              for k, v in l_dict.items()}
                    losses.update(l_dict)

            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn,
                        dn_num_boxes, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v
                              for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Replace NaN with 0 for stability
        losses = {k: torch.nan_to_num(v, nan=0.0)
                  for k, v in losses.items()}
        return losses

    # ================================================================
    # Meta info helper (original DEIM, unchanged)
    # ================================================================
    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][
            self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError(
                f"Unknown boxes_weight_format: "
                f"{self.boxes_weight_format}")

        if loss in ('boxes',):
            return {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            return {'values': iou}
        return {}

    # ================================================================
    # CDN denoising indices (original DEIM, unchanged)
    # ================================================================
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(
                    num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append(
                    (dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device),
                ))
        return dn_match_indices

    # ================================================================
    # Utility functions (original DEIM, unchanged)
    # ================================================================
    def feature_loss_function(self, fea, target_fea):
        loss = ((fea - target_fea) ** 2
                * ((fea > 0) | (target_fea > 0)).float())
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
            self, pred, label, weight_right, weight_left,
            weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = (
            F.cross_entropy(pred, dis_left, reduction='none')
            * weight_left.reshape(-1)
            + F.cross_entropy(pred, dis_right, reduction='none')
            * weight_right.reshape(-1))
        if weight is not None:
            loss = loss * weight.float()
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss

    def get_gradual_steps(self, outputs):
        num_layers = (len(outputs['aux_outputs']) + 1
                      if 'aux_outputs' in outputs else 1)
        step = .5 / (num_layers - 1)
        return ([.5 + step * i for i in range(num_layers)]
                if num_layers > 1 else [1])
