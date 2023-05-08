# # model settings
# model = dict(
#     backbone=dict(num_segments=8),
#     cls_head=dict(num_classes=51, num_segments=8))

# # dataset settings
# split = 1
# dataset_type = 'RawframeDataset'
# data_root = 'data/hmdb51/rawframes'
# data_root_val = 'data/hmdb51/rawframes'
# ann_file_train = f'data/hmdb51/hmdb51_train_split_{split}_rawframes.txt'
# ann_file_val = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
# ann_file_test = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

# train_pipeline = [
#     dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(
#         type='MultiScaleCrop',
#         input_size=224,
#         scales=(1, 0.875, 0.75, 0.66),
#         random_crop=False,
#         max_wh_scale_gap=1,
#         num_fixed_crops=13),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]
# val_pipeline = [
#     dict(
#         type='SampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]
# test_pipeline = [
#     dict(
#         type='SampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]

# data = dict(
#     videos_per_gpu=12,
#     workers_per_gpu=2,
#     test_dataloader=dict(videos_per_gpu=1),
#     train=dict(
#         type=dataset_type,
#         ann_file=ann_file_train,
#         data_prefix=data_root,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=ann_file_val,
#         data_prefix=data_root_val,
#         pipeline=val_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=ann_file_test,
#         data_prefix=data_root_val,
#         pipeline=test_pipeline))
# evaluation = dict(
#     interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# # optimizer
# optimizer = dict(
#     lr=0.0015,  # this lr is used for 8 gpus
# )
# # learning policy
# lr_config = dict(policy='step', step=[10, 20])
# total_epochs = 10 # originally 25

# load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth'  # noqa: E501
# # runtime settings
# work_dir = './work_dirs/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb/'

# model settings
model = dict(
    type='Sampler2DRecognizer2D',
    num_segments=6,
    use_sampler=True,
    bp_mode='tsn',
    explore_rate=0.1,
    resize_px=128,
    sampler=dict(
        type='MobileNetV2TSM',
        pretrained='modelzoo/anet_mobilenetv2_tsm_sampler_checkpoint.pth',
        is_sampler=True,
        shift_div=10,
        num_segments=10,
        total_segments=10),
    backbone=dict(
        type='ResNet50'),
    cls_head=dict(
        type='R50Head',
        num_classes=200,
        in_channels=2048,
        frozen=True,
        final_loss=False))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)

# dataset settings
split = 1
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = f'data/hmdb51/hmdb51_train_split_{split}_rawframes.txt'
ann_file_val = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
ann_file_test = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=40,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=80, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=80, workers_per_gpu=4),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        multi_class=True,
        num_classes=200,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        multi_class=True,
        num_classes=200,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001 / 8 * 40, momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 50
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/hmdb51_10to6_resnet50'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(
    interval=1, metrics=['mean_average_precision'], gpu_collect=True)
# directly port classification checkpoint from FrameExit
# TODO: need to be changed using frameexit
# load_from = 'modelzoo/anet_frameexit_classification_checkpoint.pth'
# load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth'  # noqa: E501
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
