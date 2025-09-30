
# local_configs/MyEdges/weighted_ce_helper.py
# Usage from a config:
#   from local_configs.MyEdges.weighted_ce_helper import apply_weighted_ce
#   apply_weighted_ce(globals(), class_weight=[1.0, 3.0, 3.0], dice_weight=0.0)
#
# This injects class-weighted CrossEntropyLoss into decode_head (+ aux head if present).

def _mk_losses(class_weight, dice_weight=0.0, ignore_index=255):
    losses = [dict(type='CrossEntropyLoss',
                   loss_weight=1.0,
                   class_weight=class_weight,
                   ignore_index=ignore_index)]
    if dice_weight and dice_weight > 0:
        losses.append(dict(type='DiceLoss',
                           loss_weight=float(dice_weight),
                           ignore_index=ignore_index))
    return losses

def apply_weighted_ce(G, class_weight, dice_weight=0.0, sampler=None):
    """Mutate the global config dicts to use weighted CE (and optional Dice).

    Args:
        G: globals() from the config file
        class_weight: list of floats, length = num_classes
        dice_weight: float, optional extra Dice loss weight
        sampler: optional dict, e.g. {'type':'OHEMPixelSampler','thresh':0.7,'min_kept':100000}
    """
    model = G['model']
    # decode head
    dh = model['decode_head']
    dh['loss_decode'] = _mk_losses(class_weight, dice_weight)
    if sampler:
        dh['sampler'] = sampler

    # auxiliary head(s) if present
    if 'auxiliary_head' in model:
        ah = model['auxiliary_head']
        if isinstance(ah, list):
            for h in ah:
                h['loss_decode'] = _mk_losses(class_weight, dice_weight)
        else:
            ah['loss_decode'] = _mk_losses(class_weight, dice_weight)
