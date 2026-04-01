"""
[MODIFIED] Detection training solver with ALTR + SPFM integration.

Changes from the original DEIM ``det_solver.py`` are marked with
``# [NEW]`` or ``# [MODIFIED]`` and correspond to:

  - **ALTR** (Adaptive Long-Tail Resampling, Section 3.4):
    Replaces the default dataloader with a class-balanced RFS loader
    whose per-class repeat factors are dynamically updated every
    evaluation epoch using validation AP feedback (Eq. 1-3).

  - **SPFM** (Self-Paced Focused Matching, Section 3.5):
    Runs a lightweight teacher evaluation on the training set to
    estimate per-class easiness scores, which are then passed to
    the criterion as priority weights with temperature scheduling.

All additions are guarded by ``ALTR.USE`` / ``SPFM.USE`` flags in the
YAML config so that the solver degrades gracefully to the DEIM baseline
when both are ``false``.

Based on DEIM (https://github.com/ShihuaHuang95/DEIM).
Modified from D-FINE (https://github.com/Peterande/D-FINE).
"""

import time
import json
import datetime
import logging
import math
import copy

import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path

# --- [NEW: ALTR] Class-balanced sampler ---
from engine.altr_sampler import (
    build_altr_loader,
    ALTRSampler,
    DistributedALTRSampler,
)
# --- [NEW: ALTR + SPFM] Dataset statistics ---
from engine.dataset_info import (
    NUM_IMAGES_PER_CLASS,
    NUM_SAMPLES_PER_CLASS,
    TOTAL_CLASSES,
)

from ..misc import dist_utils, stats, get_weight_size
from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler

logger = logging.getLogger(__name__)

# COCO metric names (for tensorboard logging)
_COCO_METRIC_NAMES = [
    'ap', 'ap50', 'ap75', 'aps', 'apm', 'apl',
    'ar', 'ar50', 'ar75', 'ars', 'arm', 'arl',
]


# ================================================================
# [NEW: SPFM]  Teacher evaluation — per-class easiness estimation
# ================================================================
def _get_true_labels(targets):
    """Extract unique class labels present in each image's targets."""
    image_labels = []
    for t in targets:
        if 'labels' in t and t['labels'].numel() > 0:
            image_labels.append(torch.unique(t['labels']).cpu().tolist())
        else:
            image_labels.append([])
    return image_labels


def _get_max_confidence(prediction, class_id):
    """Return the highest confidence score for ``class_id`` in one image."""
    if 'scores' not in prediction or prediction['scores'].numel() == 0:
        return 0.0
    mask = prediction['labels'] == class_id
    if mask.sum() == 0:
        return 0.0
    return prediction['scores'][mask].max().cpu().item()


def update_easiness_scores(model, postprocessor, eval_loader,
                           E_old_buffer, current_epoch, cfg, device):
    """
    [NEW: SPFM] Teacher evaluation for per-class easiness estimation.

    For each class c, computes the mean of the highest confidence scores
    across all training images containing class c. The raw score is then
    stabilised via an adaptive EMA whose trust coefficient increases with
    both the sample count N_c and the training progress (epoch ratio).

    This implements the "self-paced" component of SPFM (Section 3.5):
    classes that the model finds easy (high confidence) receive lower
    priority, while difficult classes receive higher priority in the
    subsequent training epochs.

    Args:
        model: detection model (switched to eval mode internally)
        postprocessor: DEIM postprocessor for decoding outputs
        eval_loader: DataLoader over the training set (no augmentation)
        E_old_buffer: list[float] of length TOTAL_CLASSES, updated in-place
        current_epoch: current training epoch
        cfg: global config object
        device: torch device
    """
    logger.info(f"[SPFM] Epoch {current_epoch}: running teacher evaluation...")
    model.eval()

    spfm_cfg = cfg.yaml_cfg.get('SPFM', {})
    ema_threshold = spfm_cfg.get('EMA_THRESHOLD', 100)
    total_epochs  = spfm_cfg.get('TOTAL_EPOCHS', 120)

    # Collect per-class max-confidence scores
    E_new_scores = [[] for _ in range(TOTAL_CLASSES)]

    with torch.no_grad():
        for images, targets in eval_loader:
            images = images.to(device)

            # Ensure consistent input resolution
            if images.shape[-1] != 640 or images.shape[-2] != 640:
                images = torch.nn.functional.interpolate(
                    images, size=(640, 640),
                    mode='bilinear', align_corners=False,
                )

            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            image_labels = _get_true_labels(targets)

            h, w = images.shape[-2:]
            orig_sizes = torch.tensor(
                [[h, w]] * len(targets),
                device=device,
            )

            outputs = model(images)
            predictions = postprocessor(outputs, orig_target_sizes=orig_sizes)

            for i in range(len(images)):
                for c in image_labels[i]:
                    c = int(c)
                    if c >= TOTAL_CLASSES:
                        continue
                    conf = _get_max_confidence(predictions[i], c)
                    E_new_scores[c].append(conf)

    # Stabilise with adaptive EMA
    for c in range(TOTAL_CLASSES):
        # Raw easiness for this epoch
        if E_new_scores[c]:
            E_c_new = float(torch.mean(torch.tensor(E_new_scores[c])))
        else:
            E_c_new = float(E_old_buffer[c])  # keep previous if no samples

        # Adaptive trust coefficient (Eq. in Section 3.5)
        # Increases with both sample count and training progress
        N_c = max(NUM_SAMPLES_PER_CLASS.get(c, 1), 1)
        alpha_samples  = min(1.0, N_c / max(ema_threshold, 1e-6))
        alpha_progress = current_epoch / max(total_epochs, 1)
        alpha_c = max(alpha_samples, alpha_progress)

        # EMA update
        E_old_buffer[c] = alpha_c * E_c_new + (1 - alpha_c) * E_old_buffer[c]

    logger.info(
        f"[SPFM] Teacher evaluation done. "
        f"E_stable (first 5): "
        f"{[round(float(e), 3) for e in E_old_buffer[:5]]}"
    )
    model.train()


# ================================================================
#  Main Solver
# ================================================================
class DetSolver(BaseSolver):
    """
    Detection solver with ALTR and SPFM integration.

    Extends the DEIM BaseSolver with two training-side modules:
    - ALTR: adaptive class-balanced resampling (Section 3.4)
    - SPFM: self-paced focused matching via teacher evaluation (Section 3.5)
    """

    def fit(self, cfg_str):
        self.train()
        args = self.cfg

        # ============================================================
        # Inject config into criterion (needed for SPFM parameters)
        # ============================================================
        if hasattr(self.criterion, 'set_cfg_externally'):
            self.criterion.set_cfg_externally(self.cfg)
            self.criterion.output_dir = self.output_dir
            logger.info("[Init] Config injected into criterion")

        # ============================================================
        # [NEW: SPFM] Build teacher evaluation loader
        # ============================================================
        self.teacher_eval_loader = None
        spfm_cfg = self.cfg.yaml_cfg.get('SPFM', {})
        if spfm_cfg.get('USE', False):
            # Use the training dataloader for teacher evaluation
            # (evaluates model confidence on training samples)
            self.teacher_eval_loader = self.train_dataloader
            logger.info("[SPFM] Teacher evaluation loader ready")

        # Per-class easiness buffer, initialised to 0.5 (neutral)
        E_old_buffer = [0.5] * TOTAL_CLASSES

        # ============================================================
        # [NEW: ALTR] Replace default loader with class-balanced loader
        # ============================================================
        altr_cfg = self.cfg.yaml_cfg.get('ALTR', {})
        if altr_cfg.get('USE', False):
            self.train_dataloader = build_altr_loader(
                self.cfg, self.train_dataloader.dataset
            )
            logger.info(
                f"[ALTR] Class-balanced dataloader built "
                f"(RFS_T = {altr_cfg['RFS_T']})"
            )

        # Snapshot original per-class image counts (before AP feedback)
        # Used as the baseline N_c in Eq. 3 to prevent count drift
        self._orig_img_counts = copy.deepcopy(NUM_IMAGES_PER_CLASS)

        # ============================================================
        # Standard DEIM setup
        # ============================================================
        if dist_utils.is_main_process():
            with open(self.output_dir / 'args.json', 'w') as f:
                f.write(cfg_str)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + " Start training " + "-" * 42)

        # Learning rate scheduler
        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer, args.lr_gamma, iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
                lr_scyedule_save_path=self.output_dir,
            )
            self.self_lr_scheduler = True

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Trainable parameters: {n_parameters:,}")

        # Evaluate if resuming from checkpoint
        best_stat = {'epoch': -1}
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device,
                output_dir=self.output_dir,
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]

        start_time = time.time()
        start_epoch = self.last_epoch + 1

        # ============================================================
        # Training loop
        # ============================================================
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            self.criterion.current_epoch = epoch

            # Propagate epoch to actual sampler (needed for DDP / ALTR)
            if dist_utils.is_dist_available_and_initialized():
                sampler = self.train_dataloader.batch_sampler.sampler
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)

            # ────────────────────────────────────────────────────
            # [NEW: SPFM] Teacher evaluation + priority update
            # ────────────────────────────────────────────────────
            if spfm_cfg.get('USE', False):
                update_interval = spfm_cfg.get('UPDATE_INTERVAL', 5)

                # Run teacher evaluation at scheduled intervals
                if (epoch % update_interval == 0
                        and self.teacher_eval_loader is not None):
                    eval_model = (self.ema.module
                                  if self.ema else self.model)
                    update_easiness_scores(
                        eval_model, self.postprocessor,
                        self.teacher_eval_loader,
                        E_old_buffer, epoch, self.cfg, self.device,
                    )

                # Pass easiness scores to criterion for priority weights
                if hasattr(self.criterion, 'update_priority_weights'):
                    self.criterion.update_priority_weights(
                        E_old_buffer, epoch
                    )

            # Handle EMA decay restart (DEIM feature)
            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.ema.decay = (
                    self.train_dataloader.collate_fn.ema_restart_decay
                )
                logger.info(
                    f"EMA decay refreshed at epoch {epoch}: {self.ema.decay}"
                )

            # ────────────────────────────────────────────────────
            # Train one epoch
            # ────────────────────────────────────────────────────
            train_stats = train_one_epoch(
                self.self_lr_scheduler, self.lr_scheduler,
                self.model, self.criterion, self.train_dataloader,
                self.optimizer, self.device, epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema, scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                plot_train_batch_freq=args.plot_train_batch_freq,
                output_dir=self.output_dir,
                epoches=args.epoches,
                verbose_type=args.verbose_type,
            )

            # LR scheduler step
            if not self.self_lr_scheduler:
                if (self.lr_warmup_scheduler is None
                        or self.lr_warmup_scheduler.finished()):
                    self.lr_scheduler.step()

            self.last_epoch += 1

            # Save checkpoints
            if self.output_dir:
                ckpt_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    ckpt_paths.append(
                        self.output_dir / f'checkpoint{epoch:04}.pth'
                    )
                for p in ckpt_paths:
                    dist_utils.save_on_master(self.state_dict(), p)

            # ────────────────────────────────────────────────────
            # Evaluate
            # ────────────────────────────────────────────────────
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device,
                output_dir=self.output_dir,
            )

            # ────────────────────────────────────────────────────
            # [NEW: ALTR] Dynamic AP feedback — update sampler
            # ────────────────────────────────────────────────────
            if altr_cfg.get('USE', False):
                self._update_altr_from_eval(coco_evaluator)

            # ────────────────────────────────────────────────────
            # [NEW: SPFM] Log adaptive focus centre μ
            # ────────────────────────────────────────────────────
            if (dist_utils.is_main_process()
                    and hasattr(self.criterion, 'adaptive_mu')):
                mu = float(self.criterion.adaptive_mu)
                probe_dir = self.output_dir / "SPFM_probes"
                probe_dir.mkdir(parents=True, exist_ok=True)
                with open(probe_dir / "spfm_mu_log.txt", "a") as f:
                    f.write(f"Epoch: {epoch}, Mu: {mu:.4f}\n")

            # ────────────────────────────────────────────────────
            # Best model tracking & logging
            # ────────────────────────────────────────────────────
            self._update_best_and_log(
                epoch, args, test_stats, train_stats,
                coco_evaluator, best_stat, n_parameters,
            )

        total_time = str(datetime.timedelta(
            seconds=int(time.time() - start_time)))
        logger.info(f"Training completed in {total_time}")

    # ================================================================
    # [NEW: ALTR]  AP feedback → sampler update  (Eq. 2-3)
    # ================================================================
    def _update_altr_from_eval(self, coco_evaluator):
        """
        Update ALTR sampling weights using per-class AP from evaluation.

        For each class c with original image count N_c^orig:
            gamma_c = clip(AP_c / A0,  gamma_min, gamma_max)    (Eq. 2)
            N_c^eff = max(1, N_c^orig * gamma_c)

        Classes with AP below the anchor A0 get gamma < 1, which
        *reduces* their effective count and thus *increases* their
        RFS repeat factor — resulting in stronger oversampling.

        The sampler is then rebuilt with the updated counts.
        """
        if coco_evaluator is None:
            return
        if 'bbox' not in coco_evaluator.coco_eval:
            return

        try:
            coco_eval = coco_evaluator.coco_eval['bbox']
            precisions = coco_eval.eval['precision']
            # Average over IoU thresholds and recall points
            # precision shape: (T, R, K, A, M)
            class_aps = np.mean(precisions[:, :, :, 0, 2], axis=(0, 1))

            sorted_cat_ids = sorted(NUM_IMAGES_PER_CLASS.keys())
            TARGET_AP = 0.65     # A0: target AP anchor (Eq. 2)

            for idx, ap in enumerate(class_aps):
                if idx >= len(sorted_cat_ids):
                    break
                cat_id = sorted_cat_ids[idx]
                original = self._orig_img_counts.get(
                    cat_id, NUM_IMAGES_PER_CLASS[cat_id]
                )

                # Density scaling factor (Eq. 2)
                # factor < 1 for underperforming classes → more sampling
                # factor > 1 for well-performing classes → less sampling
                factor = 1.0 + (ap - TARGET_AP)
                factor = max(0.5, min(1.2, factor))  # [gamma_min, gamma_max]

                NUM_IMAGES_PER_CLASS[cat_id] = max(1, int(original * factor))

            # Rebuild sampler with updated effective counts
            rfs_t = self.cfg.yaml_cfg['ALTR']['RFS_T']
            if dist_utils.is_dist_available_and_initialized():
                new_sampler = DistributedALTRSampler(
                    self.train_dataloader.dataset,
                    rfs_t=rfs_t, shuffle=True,
                )
            else:
                new_sampler = ALTRSampler(
                    self.train_dataloader.dataset,
                    rfs_t=rfs_t, shuffle=True,
                )

            # Replace sampler in existing dataloader
            self.train_dataloader.batch_sampler.sampler = new_sampler
            logger.info(
                f"[ALTR] Sampler rebuilt with "
                f"{len(new_sampler)} resampled indices"
            )

        except Exception as e:
            logger.warning(f"[ALTR] Failed to update sampler: {e}")

    # ================================================================
    # Logging & best model management (standard, lightly modified)
    # ================================================================
    def _update_best_and_log(self, epoch, args, test_stats, train_stats,
                             coco_evaluator, best_stat, n_parameters):
        """Handle best-model saving, per-class AP logging, TensorBoard."""

        for k in test_stats:
            # TensorBoard
            if self.writer and dist_utils.is_main_process():
                for i, v in enumerate(test_stats[k]):
                    self.writer.add_scalar(
                        f'Test/{k}_{_COCO_METRIC_NAMES[i]}', v, epoch
                    )

            # Track best
            prev_best = best_stat.get(k, 0)
            if test_stats[k][0] > prev_best:
                best_stat['epoch'] = epoch
                best_stat[k] = test_stats[k][0]

                logger.info(
                    f"New best at epoch {epoch}: "
                    f"AP {prev_best:.4f} -> {best_stat[k]:.4f}"
                )

                # Save best checkpoint and per-class AP report
                if self.output_dir and dist_utils.is_main_process():
                    self._save_best_results(epoch, best_stat, k,
                                            coco_evaluator)

                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    dist_utils.save_on_master(
                        self.state_dict(),
                        self.output_dir / 'best_stg2.pth',
                    )
                else:
                    dist_utils.save_on_master(
                        self.state_dict(),
                        self.output_dir / 'best_stg1.pth',
                    )
            elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                # Slightly reduce EMA decay if no improvement
                self.ema.decay -= 0.0001

            logger.info(f"best_stat: {best_stat}")

        # Write training log
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Save COCO eval state
        if coco_evaluator is not None:
            (self.output_dir / 'eval').mkdir(exist_ok=True)
            if "bbox" in coco_evaluator.coco_eval:
                filenames = ['latest.pth']
                if epoch % 50 == 0:
                    filenames.append(f'{epoch:03}.pth')
                for name in filenames:
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        self.output_dir / "eval" / name,
                    )

    def _save_best_results(self, epoch, best_stat, metric_key,
                           coco_evaluator):
        """Save best predictions JSON and per-class AP text report."""
        if coco_evaluator is None or "bbox" not in coco_evaluator.coco_eval:
            return
        try:
            coco_eval = coco_evaluator.coco_eval['bbox']

            # Save prediction JSON
            predictions = coco_eval.cocoDt.dataset.get('annotations', [])
            if predictions:
                save_path = self.output_dir / "eval" / "pred_best.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(predictions, f)

            # Per-class AP report
            precision = coco_eval.eval['precision']
            cat_ids = coco_eval.params.catIds
            dataset = self.train_dataloader.dataset
            cat_map = (dataset.category2name
                       if hasattr(dataset, 'category2name')
                       else {cid: str(cid) for cid in cat_ids})

            lines = [
                "Per-Category AP @ IoU=0.50:0.95 | area=all | maxDets=100",
                f"Best Epoch: {epoch} "
                f"(mAP: {best_stat[metric_key]:.5f})",
                "---",
            ]
            for k_idx, cat_id in enumerate(cat_ids):
                p = precision[:, :, k_idx, 0, 2]
                valid = p[p > -1]
                ap_val = float(np.mean(valid)) if valid.size > 0 else 0.0

                name = (cat_map[cat_id]
                        if isinstance(cat_map, dict)
                        else (cat_map[cat_id]
                              if cat_id < len(cat_map)
                              else str(cat_id)))
                lines.append(f"[{name}] (ID {cat_id}): {ap_val:.5f}")

            txt_path = self.output_dir / "best_per_class_ap.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            logger.info(f"Best per-class AP saved to {txt_path}")

        except Exception as e:
            logger.warning(f"Failed to save best results: {e}")

    # ================================================================
    # Evaluation entry point
    # ================================================================
    def val(self):
        """Run standalone evaluation with fused model."""
        self.eval()

        module = self.ema.module if self.ema else self.model
        module.deploy()
        _, model_info = stats(self.cfg, module=module)
        logger.info(f"Model info (fused): {model_info}")

        get_weight_size(module)
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            self.val_dataloader, self.evaluator, self.device,
            True, self.output_dir,
        )

        if (self.output_dir and dist_utils.is_main_process()
                and coco_evaluator is not None
                and "bbox" in coco_evaluator.coco_eval):
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                self.output_dir / "eval.pth",
            )
