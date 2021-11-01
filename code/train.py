from pytorch_lightning import profiler
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn.modules.module import T
from pytorch_lightning.loggers import WandbLogger
import wandb

from data import DataModule
from model import VisualModel
from vit import ViT

class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar

class WandbImagePredCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices."""
    
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples['sample'], val_samples['label']
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)
        trainer.logger.experiment.log({
            "val/examples": [
                wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
            "global_step": trainer.global_step
            })

def main():
    input_shape = (3, 224, 224)
    classes = 3
    train_imgpath = "data/images"
    train_csvpath = "data/devset.csv"
    val_imgpath = "data/test/images"
    val_csvpath = "data/testset_mediaeval.csv"
    all_labels = {'negative':0,'neutral': 1,'positive':2}
    # all_labels = {'negative':0,'positive':1}
    pl.seed_everything(42)

    class_data = DataModule(input_shape[1:],  train_imgpath, val_imgpath, train_csvpath, val_csvpath,  task=1, batch_size=16)
    class_model = VisualModel(input_shape, classes, all_labels)
    class_data.prepare_data()    
    class_data.setup()
    val_samples = next(iter(class_data.val_dataloader()))

    bar = LitProgressBar()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=7, verbose=True, mode="min"
    )
    wandb_logger = WandbLogger(name="Resnet101_bce_im", project='mediaeval21_visualsentiment', id='Resnet101_bce_im', job_type='train')
    wandb_logger.watch(class_model)

    trainer = pl.Trainer(
        default_root_dir="logs",
        profiler=True,
        progress_bar_refresh_rate=10,
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=200,
        fast_dev_run=False,
        # logger=pl.loggers.TensorBoardLogger("logs/", name="image", version=1),
        logger = wandb_logger,
        check_val_every_n_epoch=5,
        callbacks=[bar],
    )
    trainer.fit(class_model, class_data)
    # trainer.test(datamodule=class_data)
    # wandb.finish()


if __name__ == "__main__":
    main()
