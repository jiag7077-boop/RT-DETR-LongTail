# Data

## In-house Industrial Defect Dataset

The in-house power line defect dataset is proprietary and cannot be
publicly released. It contains 5,459 annotated instances across 7
categories with a long-tailed distribution.

Please refer to Section 4.1.1 of the paper for detailed statistics,
and update `engine/dataset_info.py` if adapting to your own dataset.

## DeepPCB

1. Download the original dataset:
   https://github.com/tangsanli5201/DeepPCB

2. Construct the long-tailed training set:
   ```bash
   python tools/construct_deeppcb_longtail.py \
       --data_root /path/to/DeepPCB \
       --output_dir /path/to/DeepPCB_LongTail \
       --imbalance_factor 10