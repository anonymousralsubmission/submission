# Auto-generated DFormerv2 config for 3 classes\n
from local_configs.NYUDepthv2.DFormerv2_Base import *  # noqa

CLASSES = ('corners', 'outer_edges', 'inner_edges')
PALETTE = [(0, 255, 255), (255, 0, 0), (0, 255, 0)]
num_classes = 3
metainfo = dict(classes=CLASSES, palette=PALETTE)

data_root = 'datasets/MyEdges'
for _loader in [train_dataloader, val_dataloader, test_dataloader]:
    ds = _loader.get('dataset', _loader)
    ds['data_root'] = data_root
    ds['metainfo'] = metainfo
    if 'data_prefix' in ds and isinstance(ds['data_prefix'], dict):
        ds['data_prefix'].setdefault('img_path', 'RGB')
        ds['data_prefix'].setdefault('depth_path', 'Depth')
        ds['data_prefix'].setdefault('seg_map_path', 'Labels')
    ds.setdefault('ignore_index', 255)
    ds['reduce_zero_label'] = False

model['decode_head']['num_classes'] = num_classes
if 'auxiliary_head' in model:
    if isinstance(model['auxiliary_head'], list):
        for aux in model['auxiliary_head']:
            aux['num_classes'] = num_classes
    else:
        model['auxiliary_head']['num_classes'] = num_classes

if 'loss_decode' in model['decode_head']:
    model['decode_head']['loss_decode']['ignore_index'] = 255
else:
    model['decode_head']['loss_decode'] = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, ignore_index=255)
if 'auxiliary_head' in model:
    if isinstance(model['auxiliary_head'], list):
        for aux in model['auxiliary_head']:
            if 'loss_decode' in aux:
                aux['loss_decode']['ignore_index'] = 255
    else:
        if 'loss_decode' in model['auxiliary_head']:
            model['auxiliary_head']['loss_decode']['ignore_index'] = 255

load_from = 'checkpoints/pretrained/DFormerv2_Base_pretrained.pth'
work_dir = './checkpoints/MyEdges/DFormerv2_Base_3cls'
