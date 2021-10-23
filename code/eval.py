from pytorch_lightning import profiler
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn.modules.module import T
from pytorch_lightning.loggers import WandbLogger, wandb

from data import DataModule
from model import VisualModel
from vit import ViT

class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar

def main():
    input_shape = (3, 224, 224)
    classes = 3
    imgpath = "data/images"
    csvpath = "data/devset.csv"
    all_labels = {'negative':0,'neutral':1,'positive':2}

    class_data = DataModule(input_shape[1:], imgpath, csvpath, task=1, batch_size=32)
    class_model = VisualModel(input_shape, classes, all_labels).load_from_checkpoint("wandb/run-20211023_121752-Resnet101_CE/files/mediaeval21_visualsentiment/Resnet101_CE/checkpoints/epoch=1-step=303.ckpt")
    # class_model.freeze()
    bar = LitProgressBar()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )
    # wandb_logger = WandbLogger(name="twinssvt", project='mediaeval21_visualsentiment', job_type='train')
    # wandb_logger.watch(class_model)

    trainer = pl.Trainer(
        default_root_dir="logs",
        profiler=True,
        progress_bar_refresh_rate=4,
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=100,
        fast_dev_run=False,
        callbacks=[bar],
    )
    # trainer.test(ckpt_path="wandb/run-20211007_183158-10wqqobr/files/mediaeval21_visualsentiment/10wqqobr/checkpoints/epoch=13-step=1707.ckpt",test_dataloaders=class_data.test_dataloader())
    trainer.test(class_model,datamodule=class_data)
    # print(class_model.temp)
    # wandb.finish()


if __name__ == "__main__":
    main()
