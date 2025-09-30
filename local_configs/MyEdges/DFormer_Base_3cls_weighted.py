
# local_configs/MyEdges/DFormer_Base_3cls_weighted.py
# DFormer (not v2) 3-class config with class-weighted CE (and optional Dice).
from local_configs.NYUDepthv2.DFormer_Base import *  # noqa
from local_configs.MyEdges.common_loader_3cls import patch_for_3cls
from local_configs.MyEdges.weighted_ce_helper import apply_weighted_ce

# dataset & pretrained (reuse your paths as needed)
_pretrained = 'checkpoints/pretrained/DFormer_Base_pretrained.pth'
patch_for_3cls(globals(), data_root='datasets/MyEdges', pretrained=_pretrained)

# ---- weights: EDIT THESE three numbers to your dataset ----
CLASS_WEIGHT = [1.0, 3.0, 3.0]  # [corners, outer_edges, inner_edges]
DICE_WEIGHT = 0.0               # set >0 (e.g., 3.0) to also add Dice

# optional hard example mining
SAMPLER = dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)

apply_weighted_ce(globals(), class_weight=CLASS_WEIGHT, dice_weight=DICE_WEIGHT, sampler=SAMPLER)

work_dir = './checkpoints/MyEdges/DFormer_Base_3cls_weighted'
