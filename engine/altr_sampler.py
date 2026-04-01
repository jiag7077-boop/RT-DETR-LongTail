"""
ALTR (Adaptive Long-Tail Resampling) Sampler — Section 3.4

Implements Repeat Factor Sampling (RFS) with dynamically updated per-class
sampling rates based on validation AP feedback from the solver.

Key equations:
  r_c = max(1, sqrt(T / N_c))                           (Eq. 1)
  gamma_c = clip(AP_c / A0,  gamma_min, gamma_max)      (Eq. 2)
  r_c^t = max(1, sqrt(T / (N_c * gamma_c^t)))           (Eq. 3)

The solver (det_solver.py) updates NUM_IMAGES_PER_CLASS after each
validation epoch, and the sampler rebuilds its index list accordingly.

Based on DEIM (https://github.com/ShihuaHuang95/DEIM).
"""

import math
import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler, RandomSampler, DistributedSampler

from .dataset_info import NUM_IMAGES_PER_CLASS, TOTAL_CLASSES
from .misc import dist_utils

logger = logging.getLogger(__name__)


class ALTRSampler(Sampler):
    """
    RFS-based class-balanced sampler for long-tail detection.

    Computes per-class repeat factors as:
        r_c = max(1, sqrt(T / N_c))
    where T is a frequency threshold and N_c is the effective sample count
    for class c. N_c is dynamically updated by the solver via ALTR's
    validation AP feedback loop (see det_solver.py).

    Args:
        dataset: COCO-style detection dataset with .coco and .ids attributes
        rfs_t (float): frequency threshold T for RFS
        seed (int): random seed for reproducibility
        shuffle (bool): whether to shuffle indices each epoch
    """

    def __init__(self, dataset, rfs_t, seed=0, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.rfs_t = rfs_t
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

        self.cat_ids = sorted(NUM_IMAGES_PER_CLASS.keys())
        if not self.cat_ids:
            raise ValueError("NUM_IMAGES_PER_CLASS is empty")

        # Compute initial repeat factors (Eq. 1)
        self.repeat_factors = {}
        for c in self.cat_ids:
            Nc = NUM_IMAGES_PER_CLASS.get(c, 0)
            r_c = math.sqrt(self.rfs_t / Nc) if Nc > 0 else 0
            self.repeat_factors[c] = max(1.0, r_c)

        # Build per-class image index mapping
        self.cat_id_to_img_indices = {}
        for c in self.cat_ids:
            coco_img_ids = set(self.dataset.coco.getImgIds(catIds=c))
            self.cat_id_to_img_indices[c] = [
                i for i, img_id in enumerate(self.dataset.ids)
                if img_id in coco_img_ids
            ]

        self.indices = self._build_sample_indices()
        self.num_samples = len(self.indices)

        logger.info(
            f"ALTR sampler: RFS T={self.rfs_t}, "
            f"original={len(self.dataset)}, resampled={self.num_samples}")

    def _build_sample_indices(self):
        """Build oversampled index list. Integer part fully repeated,
        fractional part randomly sampled."""
        final_indices = []
        for c in self.cat_ids:
            img_indices = self.cat_id_to_img_indices[c]
            if not img_indices:
                continue
            rf = self.repeat_factors[c]
            num_full = int(math.floor(rf))
            remainder = rf - num_full

            for _ in range(num_full):
                final_indices.extend(img_indices)

            if remainder > 0:
                num_extra = int(math.ceil(len(img_indices) * remainder))
                rng = np.random.RandomState(self.seed + self.epoch)
                extra = rng.choice(img_indices, size=num_extra,
                                   replace=False).tolist()
                final_indices.extend(extra)
        return final_indices

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.g.manual_seed(self.seed + self.epoch)
        self.indices = self._build_sample_indices()
        self.num_samples = len(self.indices)

    def __iter__(self):
        if self.shuffle:
            idx = torch.tensor(self.indices, dtype=torch.long)
            perm = torch.randperm(self.num_samples, generator=self.g)
            return iter(idx[perm].tolist())
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class DistributedALTRSampler(ALTRSampler):
    """Multi-GPU (DDP) compatible version of ALTRSampler."""

    def __init__(self, dataset, rfs_t, seed=0, shuffle=True):
        if not dist_utils.is_dist_available_and_initialized():
            raise RuntimeError("DDP not initialized")
        super().__init__(dataset, rfs_t, seed, shuffle)
        self.num_replicas = dist_utils.get_world_size()
        self.rank = dist_utils.get_rank()
        self.num_samples = int(
            math.ceil(len(self.indices) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            self.g.manual_seed(self.seed + self.epoch)
            idx = torch.tensor(self.indices, dtype=torch.long)
            shuffled = idx[torch.randperm(len(self.indices), generator=self.g)]
        else:
            shuffled = torch.tensor(self.indices, dtype=torch.long)

        padding = self.total_size - len(shuffled)
        if padding > 0:
            shuffled = torch.cat((shuffled, shuffled[:padding]))

        sub = shuffled[self.rank:self.total_size:self.num_replicas]
        return iter(sub.tolist())

    def __len__(self):
        return self.num_samples


# ==================================================================
# Builder function — called by det_solver.py
# ==================================================================
try:
    from .dataloader import DataLoader
except ImportError:
    from torch.utils.data import DataLoader


def build_altr_loader(cfg, dataset):
    """
    Build a DataLoader with ALTR sampler.

    Called once at training start by det_solver.py. The sampler's effective
    counts (NUM_IMAGES_PER_CLASS) are updated each epoch by the ALTR
    validation AP feedback loop.

    Args:
        cfg: training config object (must contain cfg.yaml_cfg['ALTR'])
        dataset: COCO-style detection dataset

    Returns:
        DataLoader with class-balanced sampling
    """
    altr_cfg = cfg.yaml_cfg.get('ALTR', {})

    if altr_cfg.get('USE', False):
        rfs_t = altr_cfg['RFS_T']
        if dist.is_initialized():
            sampler = DistributedALTRSampler(dataset, rfs_t=rfs_t, shuffle=True)
        else:
            sampler = ALTRSampler(dataset, rfs_t=rfs_t, shuffle=True)
    else:
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset, num_replicas=dist_utils.get_world_size(),
                rank=dist_utils.get_rank(), shuffle=True)
        else:
            sampler = RandomSampler(dataset)

    batch_sampler = BatchSampler(
        sampler=sampler,
        batch_size=cfg.yaml_cfg['train_dataloader']['total_batch_size'],
        drop_last=True)

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=cfg.train_dataloader.collate_fn,
        num_workers=cfg.yaml_cfg['train_dataloader']['num_workers'],
        pin_memory=True,
        shuffle=False)

    return loader