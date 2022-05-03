from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, inference_segmentor, show_result_pyplot
from mmseg.datasets import build_dataset
from mmseg.apis import set_random_seed
import mmcv
from mmcv import Config
import os.path as osp
import matplotlib.pyplot as plt
classes = ('BG', 'WT', 'TC', 'ET')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34]]
def main():

    # cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')
    cfg = Config.fromfile('configs/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_brats.py')
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = 'checkpoints/fcn_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_201821-c314da8a.pth'
    cfg.load_from = 'work_dirs/tutorial/latest.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './work_dirs/tutorial'

    cfg.runner.max_iters = 1000
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector

    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                    meta=dict())



if __name__ == '__main__':
    main()
