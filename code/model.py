import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
# from transformers import AutoModel
from sklearn.metrics import accuracy_score

import torchvision.models as models
import torchmetrics
from vit_pytorch import ViT
from vit_pytorch.levit import LeViT
from vit_pytorch.twins_svt import TwinsSVT
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
import wandb


class VisualModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, all_labels, learning_rate=0.001):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.vit = ViT(
            image_size = 224,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.levit = LeViT(
            image_size = 224,
            num_classes = num_classes,
            stages = 3,             # number of stages
            dim = (256, 384, 512),  # dimensions at each stage
            depth = 4,              # transformer of depth 4 at each stage
            heads = (4, 6, 8),      # heads at each stage
            mlp_mult = 2,
            dropout = 0.1
        )
        
        self.twinssvt = TwinsSVT(
            num_classes = num_classes,       # number of output classes
            s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
            s1_patch_size = 4,        # stage 1 - patch size for patch embedding
            s1_local_patch_size = 7,  # stage 1 - patch size for local attention
            s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
            s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
            s2_emb_dim = 128,         # stage 2 (same as above)
            s2_patch_size = 2,
            s2_local_patch_size = 7,
            s2_global_k = 7,
            s2_depth = 1,
            s3_emb_dim = 256,         # stage 3 (same as above)
            s3_patch_size = 2,
            s3_local_patch_size = 7,
            s3_global_k = 7,
            s3_depth = 5,
            s4_emb_dim = 512,         # stage 4 (same as above)
            s4_patch_size = 2,
            s4_local_patch_size = 7,
            s4_global_k = 7,
            s4_depth = 4,
            peg_kernel_size = 3,      # positional encoding generator kernel size
            dropout = 0.              # dropout
        )
        ### resnet50
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

        ### VGG16
        # backbone = models.vgg16(pretrained=True)
        # num_filters = backbone.classifier[0].in_features
        # layers = list(backbone.children())[:-1]
        # self.feature_extractor = nn.Sequential(*layers)
        # self.classifier = nn.Linear(num_filters, num_classes)


        self.predict = torch.empty((), dtype=torch.int64, device = 'cuda')
        self.all_labels = all_labels
        self.acc = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1(num_classes)
        self.conf = torchmetrics.ConfusionMatrix(num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        # import pdb; pdb.set_trace()

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # will be used during inference
    def forward(self, x):
        # x = self._forward_features(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x), dim=1)
        # x = self.twinssvt(x)
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)

        # x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        x, y = batch["sample"], batch["label"] 
        # import pdb; pdb.set_trace()
        # y = torch.reshape(y, (-1, self.num_classes))
        y_onehot = F.one_hot(y.to(torch.int64), num_classes=self.num_classes).to(torch.float32)
        logits = self(x)
        
        ### CROSS ENTROPY
        # loss = F.cross_entropy(logits, y)
        #  preds = torch.argmax(logits, dim=1)

        ### BINARY CROSS ENTROPY
        # logits = torch.sigmoid(logits)
        weight = torch.tensor([1.0, 7.0, 4.0]).to("cuda")
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot)
        loss = loss * weight
        loss = loss.mean()

        preds = logits.clone()
        preds = torch.sigmoid(preds)
        preds_f, preds_i = torch.max(preds, dim=1)
        preds_i[preds_f <= 0.8] = 1
        

        # preds[preds >= 0.6] = 1
        # preds[preds <= 0.4] = 0
        # preds[(preds > 0.4) * (preds < 0.6)] = 0.5

        f1 = self.f1(preds_i, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, logger=True)
        # self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # logits = self.forward(batch["input_ids"], batch["attention_mask"])

        x, y = batch["sample"], batch["label"] 
        # import pdb; pdb.set_trace()
        # y = torch.reshape(y, (-1, self.num_classes))
        y_onehot = F.one_hot(y.to(torch.int64), num_classes=self.num_classes).to(torch.float32)
        logits = self(x)
        
        ### CROSS ENTROPY
        # loss = F.cross_entropy(logits, y)
        #  preds = torch.argmax(logits, dim=1)

        ### BINARY CROSS ENTROPY
        # logits = torch.sigmoid(logits)
        weight = torch.tensor([1.0, 7.0, 4.0]).to("cuda")
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot)
        loss = loss * weight
        loss = loss.mean()

        preds = logits.clone()
        preds = torch.sigmoid(preds)
        preds_f, preds_i = torch.max(preds, dim=1)
        preds_i[preds_f <= 0.5] = 1
        

        # preds[preds >= 0.6] = 1
        # preds[preds <= 0.4] = 0
        # preds[(preds > 0.4) * (preds < 0.6)] = 0.5

        val_f1 = self.f1(preds_i, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)

        return {"preds": preds_i, "labels": y}

    def validation_epoch_end(self, outputs):
        
        # print("Done Validating")
        predicts = []
        labels = []

        for sample in outputs:
            predict, label = sample['preds'].tolist(), sample['labels'].tolist()
            predicts += predict
            labels += label

        predicts, labels = np.array(predicts), np.array(labels)

        # predicts[predicts == 1] = 2
        # labels[labels == 1] = 2
        # predicts[predicts == 0.5] = 1
        # labels[labels == 0.5] = 1

        predicts = predicts.astype(int)
        labels = labels.astype(int)
        # import pdb; pdb.set_trace()

        wandb.log({"Confusion matrix" : wandb.plot.confusion_matrix( 
        preds=predicts, y_true=labels,
        class_names=list(self.all_labels.keys()),
        title="ResNet50_BCE_weight")})

    def test_step(self, batch, batch_idx):
        x, y = batch["sample"], batch["label"] 
        # import pdb; pdb.set_trace()
        # y = torch.reshape(y, (-1, self.num_classes))
        y_onehot = F.one_hot(y.to(torch.int64), num_classes=self.num_classes).to(torch.float32)
        logits = self(x)
        
        ### CROSS ENTROPY
        # loss = F.cross_entropy(logits, y)
        #  preds = torch.argmax(logits, dim=1)

        ### BINARY CROSS ENTROPY
        # logits = torch.sigmoid(logits)
        weight = torch.tensor([1.0, 3.0, 2.0]).to("cuda")
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot)
        loss = loss * weight
        loss = loss.mean()

        preds = logits.clone()
        preds = torch.sigmoid(preds)
        preds_f, preds_i = torch.max(preds, dim=1)
        preds_i[preds_f <= 0.4] = 1
        

        # preds[preds >= 0.6] = 1
        # preds[preds <= 0.4] = 0
        # preds[(preds > 0.4) * (preds < 0.6)] = 0.5

        val_f1 = self.f1(preds_i, y)
        
        
        conf = {"conf" : self.conf(preds_i, y)}
        self.log_dict(conf)
        # import pdb; pdb.set_trace()
        # self.temp = val_f1
        self.log("val_f1", val_f1 , prog_bar=True)
        return {"preds": preds_i, "labels": y}
        # return self.conf(preds, y)
        
    def test_epoch_end(self, outputs):
        predicts = []
        labels = []

        for sample in outputs:
            predict, label = sample['preds'].tolist(), sample['labels'].tolist()
            predicts += predict
            labels += label
    
        cm = confusion_matrix(labels, predicts, labels=list(self.all_labels.values()))
        print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of resnet50_BCE_weight')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + list(self.all_labels.keys()))
        ax.set_yticklabels([''] + list(self.all_labels.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig("conf/resnet50_BCE_weight.png")

            
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

