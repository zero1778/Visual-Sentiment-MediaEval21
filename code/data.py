import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import cv2
import pandas as pd 

from typing import Callable

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class VisualDataset(Dataset):
    def __init__(self,  imgpath, datalist, size, task, type, transformation=True ):
        super().__init__()

        self.classes = []
        self.type = type

        self.trans = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.6),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

        ori_img = cv2.imread(self.imgpath + "/" + img_name)
        if (self.task == 1):
            label = self.class_task[self.task][self.datalist.iloc[idx][1]]
        elif (self.task == 2): 
            label = np.where(self.class_task[self.task] == self.datalist.iloc[idx][2:9])[0]
        else: 
            label = np.where(self.class_task[self.task] == self.datalist.iloc[idx][9:])[0]
        # label = torch.tensor(label)
        
        if self.transformation :
            img = self.trans[self.type](ori_img)

        # return {'sample': img,
        #         'label' : label,
        #         'img': ori_img}
        return {'sample': img,
                'label' : label}

    def get_labels(self): 
        return list(self.datalist['T1'])
        

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
            train, val = train_test_split(datalist, test_size=0.2, random_state=2021, 
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
            self.train_data, num_workers=4, 
            # sampler=ImbalancedDatasetSampler(self.train_data),
            batch_size=self.batch_size, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, num_workers=4, 
            batch_size=self.batch_size, 
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_data, num_workers=4, 
            batch_size=self.batch_size, 
            shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
