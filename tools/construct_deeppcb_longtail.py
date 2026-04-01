"""
Construct a long-tailed training set from DeepPCB.

Subsamples training images by category frequency with a given imbalance
factor, following the retention ratios in Table A2 of the paper.

Usage:
    python tools/construct_deeppcb_longtail.py \
        --data_root /path/to/DeepPCB \
        --output_dir /path/to/DeepPCB_LongTail \
        --imbalance_factor 10 \
        --seed 42
"""

import argparse
import json
import os
import random
import shutil
import math
from pathlib import Path
from collections import defaultdict


def load_coco_annotations(ann_path):
    with open(ann_path, 'r') as f:
        return json.load(f)


def construct_longtail(data_root, output_dir, imbalance_factor=10, seed=42):
    """
    Construct long-tailed version of DeepPCB training set.

    The class with the most instances retains 100% of its samples,
    while the class with the fewest retains 1/imbalance_factor.
    Intermediate classes are subsampled with exponential decay.

    Validation and test splits are kept unchanged.
    """
    random.seed(seed)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training annotations
    train_ann_path = data_root / 'train' / 'annotations.json'
    if not train_ann_path.exists():
        # Try alternative paths
        for alt in ['annotations/train.json',
                    'train/instances_train.json']:
            alt_path = data_root / alt
            if alt_path.exists():
                train_ann_path = alt_path
                break

    coco_data = load_coco_annotations(train_ann_path)

    # Count instances per category
    cat_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        cat_counts[ann['category_id']] += 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_cats[0][1]
    num_cats = len(sorted_cats)

    # Compute retention ratios with exponential decay
    retention_ratios = {}
    for rank, (cat_id, count) in enumerate(sorted_cats):
        # Exponential decay: ratio = IF^(-rank/(num_cats-1))
        if num_cats > 1:
            exponent = rank / (num_cats - 1)
        else:
            exponent = 0
        ratio = imbalance_factor ** (-exponent)
        retention_ratios[cat_id] = ratio

    print("Retention ratios:")
    for cat_id, ratio in retention_ratios.items():
        orig = cat_counts[cat_id]
        kept = int(orig * ratio)
        print(f"  Category {cat_id}: {orig} -> {kept} "
              f"({ratio:.1%} retention)")

    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # For each annotation, decide whether to keep based on its category
    # An image is kept if at least one of its annotations is retained
    kept_ann_ids = set()
    kept_img_ids = set()

    cat_ann_ids = defaultdict(list)
    for ann in coco_data['annotations']:
        cat_ann_ids[ann['category_id']].append(ann['id'])

    for cat_id, ann_ids in cat_ann_ids.items():
        ratio = retention_ratios[cat_id]
        num_keep = max(1, int(len(ann_ids) * ratio))
        random.shuffle(ann_ids)
        for aid in ann_ids[:num_keep]:
            kept_ann_ids.add(aid)

    # Keep images that have at least one retained annotation
    new_annotations = []
    for ann in coco_data['annotations']:
        if ann['id'] in kept_ann_ids:
            new_annotations.append(ann)
            kept_img_ids.add(ann['image_id'])

    new_images = [img for img in coco_data['images']
                  if img['id'] in kept_img_ids]

    # Build new annotation file
    new_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data['categories'],
    }
    if 'info' in coco_data:
        new_coco['info'] = coco_data['info']

    # Save
    out_ann_dir = output_dir / 'train'
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path = out_ann_dir / 'annotations.json'
    with open(out_ann_path, 'w') as f:
        json.dump(new_coco, f)

    # Copy val/test unchanged
    for split in ['val', 'test']:
        src = data_root / split
        dst = output_dir / split
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))

    # Summary
    new_cat_counts = defaultdict(int)
    for ann in new_annotations:
        new_cat_counts[ann['category_id']] += 1

    print(f"\nLong-tailed training set constructed at {output_dir}")
    print(f"Imbalance factor: {imbalance_factor}")
    print(f"Images: {len(coco_data['images'])} -> {len(new_images)}")
    print(f"Annotations: {len(coco_data['annotations'])} "
          f"-> {len(new_annotations)}")
    print("\nPer-category breakdown:")
    for cat_id, orig in sorted(cat_counts.items()):
        new = new_cat_counts.get(cat_id, 0)
        print(f"  Cat {cat_id}: {orig} -> {new} "
              f"({new/orig:.1%} retained)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Construct long-tailed DeepPCB training set')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to original DeepPCB dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output path for long-tailed version')
    parser.add_argument('--imbalance_factor', type=int, default=10,
                        help='Imbalance factor (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    construct_longtail(args.data_root, args.output_dir,
                       args.imbalance_factor, args.seed)