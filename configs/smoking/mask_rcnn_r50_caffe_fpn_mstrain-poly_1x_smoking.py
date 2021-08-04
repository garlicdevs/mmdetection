# The new config inherits a base config to highlight the necessary modification
_base_ = '/home/thanh/work/scanner_dl/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Cigarette',)
data = dict(
    train=dict(
        img_prefix='/home/thanh/work/scanner_dl/mmdetection/data/smoking/train/',
        classes=classes,
        ann_file='/home/thanh/work/scanner_dl/mmdetection/data/smoking/train/train_coco.json'),
    val=dict(
        img_prefix='/home/thanh/work/scanner_dl/mmdetection/data/smoking/val/',
        classes=classes,
        ann_file='/home/thanh/work/scanner_dl/mmdetection/data/smoking/val/val_coco.json'),
    test=dict(
        img_prefix='/home/thanh/work/scanner_dl/mmdetection/data/smoking/test/',
        classes=classes,
        ann_file='/home/thanh/work/scanner_dl/mmdetection/data/smoking/test/test_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'