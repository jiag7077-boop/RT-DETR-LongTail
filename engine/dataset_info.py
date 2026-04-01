"""
Dataset configuration for ALTR and SPFM modules.

This file defines per-class statistics used by:
- ALTR sampler: NUM_IMAGES_PER_CLASS for RFS repeat factors (Eq. 1-3)
- SPFM criterion: NUM_SAMPLES_PER_CLASS for priority weight computation

NUM_IMAGES_PER_CLASS is dynamically updated during training by the ALTR
validation AP feedback loop in det_solver.py (Section 3.4).

To adapt this code to a different dataset:
1. Set TOTAL_CLASSES to your number of categories
2. Update NUM_SAMPLES_PER_CLASS with per-class instance counts
3. Update NUM_IMAGES_PER_CLASS with per-class image counts
4. Update CATEGORY_NAMES with your category labels
"""

# Total number of object categories
TOTAL_CLASSES = 7

# Per-class instance counts in the training set
# Used by SPFM for teacher evaluation and priority weight computation
NUM_SAMPLES_PER_CLASS = {
    0: 11877,   # insulator              (head)
    1: 764,     # insulator_stringdrop   (medium)
    2: 1003,    # insulator_breakage     (medium)
    3: 1513,    # insulator_flashover    (medium)
    4: 1667,    # damper                 (head)
    5: 705,     # damper_defect          (medium)
    6: 271,     # nest                   (tail)
}

# Per-class image counts in the training set
# Used by ALTR sampler to compute RFS repeat factors
# Dynamically updated by solver each epoch via validation AP feedback
NUM_IMAGES_PER_CLASS = {
    0: 4163,    # insulator
    1: 679,     # insulator_stringdrop
    2: 745,     # insulator_breakage
    3: 687,     # insulator_flashover
    4: 671,     # damper
    5: 610,     # damper_defect
    6: 265,     # nest
}

# Category names indexed by class ID
CATEGORY_NAMES = [
    "insulator",
    "insulator_stringdrop",
    "insulator_breakage",
    "insulator_flashover",
    "damper",
    "damper_defect",
    "nest",
]