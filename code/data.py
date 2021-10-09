import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import cv2
import pandas as pd 

class VisualDataset(Dataset):
    def __init__(self,  imgpath, datalist, size, task, type, transformation=True ):
        super().__init__()

        self.classes = []
        self.type = type

        self.trans = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

       
        self.transformation = transformation

        self.datalist = datalist
        self.imgpath = imgpath
        self.task = task
        self.class_task = {
            1: {'negative': 0, 'neutral': 1, 'positive':2},
            2: [1,1,1,1,1,1,1], 
            3: [1,1,1,1,1,1,1,1,1]
        }
        

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        img_name = self.datalist.iloc[idx][0] + ".jpg"

        img = cv2.imread(self.imgpath + "/" + img_name)
        if (self.task == 1):
            label = self.class_task[self.task][self.datalist.iloc[idx][1]]
        elif (self.task == 2): 
            label = np.where(self.class_task[self.task] == self.datalist.iloc[idx][2:9])[0]
        else: 
            label = np.where(self.class_task[self.task] == self.datalist.iloc[idx][9:])[0]
        label = torch.tensor(label)
        
        if self.transformation:
            img = self.trans[self.type](img)

        return {'sample': img,
                'label' : label}
        

class DataModule(pl.LightningDataModule):
    def __init__(self, size, imgpath, csvpath, task, batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.size = size 
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.task = task
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        # set up class Dataset
        datalist = pd.read_csv(self.csvpath)

        if self.task == 1:
            train, val = train_test_split(datalist, test_size=0.2, random_state=140720, 
                                            stratify=datalist['T1'])

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        
        # TODO: Imbalancing processing

        self.train_data = VisualDataset(self.imgpath, train, self.size, self.task, 'train')
        self.val_data = VisualDataset(self.imgpath, val, self.size, self.task, 'val')


    def setup(self, stage=None):
        # transform, split,...

        # we set up only relevant datasets when stage is specified
        pass
        # if stage == "fit" or stage is None:
            # self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            # self.train_data.set_format(
            #     type="torch", columns=["input_ids", "attention_mask", "label"]
            # )

            # self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            # self.val_data.set_format(
            #     type="torch", columns=["input_ids", "attention_mask", "label"]
            # )
            # self.train_data = 

    def train_dataloader(self):
        return DataLoader(
            self.train_data, num_workers=4, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
    
        return DataLoader(
            self.val_data, num_workers=4, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        # print(self.val_data)
        # print(1111111111111111)
        # datalist = pd.read_csv(self.csvpath)

        # if self.task == 1:
        #     _, test = train_test_split(datalist, test_size=0.2, random_state=140720, 
        #                                     stratify=datalist['T1'])

        # # train = train.reset_index(drop=True)
        # test = test.reset_index(drop=True)
        # self.test_data = VisualDataset(self.imgpath, test, self.size, self.task)
        return DataLoader(
            self.val_data, num_workers=4, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
