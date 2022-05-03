_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16_brats.py', '../_base_/datasets/brats.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(test_cfg=dict(crop_size=(224, 224), stride=(170, 170)))
evaluation = dict(metric='mDice')
