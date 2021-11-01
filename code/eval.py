from pytorch_lightning import profiler
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn.modules.module import T
from pytorch_lightning.loggers import WandbLogger, wandb
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
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
    train_imgpath = "data/images"
    train_csvpath = "data/devset.csv"
    val_imgpath = "data/test/images"
    val_csvpath = "data/testset_mediaeval.csv"
    all_labels = {'negative':0,'neutral': 1,'positive':2}
    pl.seed_everything(42)
    # all_labels = {'negative':0,'positive':1}

    class_data = DataModule(input_shape[1:],  train_imgpath, val_imgpath, train_csvpath, val_csvpath,  task=1, batch_size=8)
    class_model = VisualModel(input_shape, classes, all_labels).load_from_checkpoint("wandb/run-20211031_072622-Resnet101_bce_im/files/mediaeval21_visualsentiment/Resnet101_bce_im/checkpoints/epoch=34-step=5319.ckpt")
    class_model.eval().cuda(device=0)
    # class_model.freeze()
    class_data.prepare_data()    
    class_data.setup()
    test_data = class_data.test_dataloader()
    # bar = LitProgressBar()

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="./models", monitor="val_loss", mode="min"
    # )
    # early_stopping_callback = EarlyStopping(
    #     monitor="val_loss", patience=5, verbose=True, mode="min"
    # )
    # wandb_logger = WandbLogger(name="twinssvt", project='mediaeval21_visualsentiment', job_type='train')
    # wandb_logger.watch(class_model)
# 
    # trainer = pl.Trainer(gpus=1)
    # trainer.test(ckpt_path="wandb/run-20211007_183158-10wqqobr/files/mediaeval21_visualsentiment/10wqqobr/checkpoints/epoch=13-step=1707.ckpt",test_dataloaders=class_data.test_dataloader())
    # trainer.test(class_model,datamodule=class_data)
    predicts = []
    labels = []
    f1_f = 0
    for batch in tqdm(test_data):
        x, y = batch["sample"].cuda(device=0), batch["label"].cuda(device=0)
        logits = class_model(x)
        preds = logits.clone()
        # import pdb;pdb.set_trace()
        preds = torch.sigmoid(preds)
        preds_f, preds_i = torch.max(preds, dim=1)
        # import pdb;pdb.set_trace()
        # preds_i[preds_f <= 0.4] = 1
        
        predict, label = preds_i.tolist(), y.tolist()
        predicts += predict
        labels += label

    f1_f =  f1_score(labels, predicts, average='weighted')
    cm = confusion_matrix(labels, predicts, labels=list(all_labels.values()))

    print(cm)
    print(f1_f)
    # print(class_model.temp)
    # wandb.finish()


if __name__ == "__main__":
    main()
