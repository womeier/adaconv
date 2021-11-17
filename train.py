from argparse import ArgumentParser
from pathlib import Path

import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from lib import dataset
from lib.lightning.datamodule import DataModule
from lib.lightning.lightningmodel import LightningModel


class CSVImageLogger(CSVLogger):
    """
    Wrapper for CSVLogger to enable logging of images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp = self.experiment

        if not hasattr(exp, 'add_image'):
            exp.add_image = self.add_image

    def add_image(self, tag, img_tensor, global_step):
        dir = Path(self.log_dir, 'images')
        dir.mkdir(parents=True, exist_ok=True)

        file = dir.joinpath(f'{tag}_{global_step:09}.jpg')
        dataset.save(img_tensor, file)


class TensorboardImageLogger(TensorBoardLogger):
    """
    Wrapper for CSVLogger to enable logging of images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp = self.experiment

        # if not hasattr(exp, 'add_image'):
        exp.add_image = self.add_image

    def add_image(self, tag, img_tensor, global_step):
        dir = Path(self.log_dir, 'images')
        dir.mkdir(parents=True, exist_ok=True)

        file = dir.joinpath(f'{tag}_{global_step:09}.jpg')
        dataset.save(img_tensor, file)


def parse_args():
    # Init parser
    parser = ArgumentParser()
    parser.add_argument('--iterations', type=int, default=160_000)
    parser.add_argument('--log-dir', type=str, default='./')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--val-interval', type=int, default=250)

    parser = DataModule.add_argparse_args(parser)
    parser = LightningModel.add_argparse_args(parser)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    args['checkpoint'] = r'logs\epoch=0-step=73999.ckpt'

    if args['checkpoint'] is None:
        model = LightningModel(**args)
    else:
        model = LightningModel.load_from_checkpoint(checkpoint_path=args['checkpoint'])
    datamodule = DataModule(**args)

    # logger = CSVImageLogger(args['log_dir'], name='logs')
    logger = TensorboardImageLogger(args['log_dir'], name='logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(gpus=1,
                      resume_from_checkpoint=args['checkpoint'],
                      max_epochs=5,
                      max_steps=args['iterations'],
                      checkpoint_callback=True,
                      val_check_interval=args['val_interval'],
                      logger=logger,
                      callbacks=[lr_monitor])

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint("./model.ckpt")
