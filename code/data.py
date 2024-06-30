import os, torch, torchvision
import pandas as pd
import pytorch_lightning as pl
import sys
sys.path.append('../')
from RandStainNA.randstainna import RandStainNA
from functools import partial
import pickle
from .utils import *

class MyRotationTransform:
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = self.angles[torch.randint(0, len(self.angles), (1,)).numpy()[0]]
        return torchvision.transforms.functional.rotate(x, angle)

def get_training_transform(horizontal_flip = True, vertical_flop = True, rotaions = True , augmentations = 'RandStainNA', normalize = True):
    training_transform = []
    if horizontal_flip:
        training_transform.append(torchvision.transforms.RandomHorizontalFlip())
    if vertical_flop:
        training_transform.append(torchvision.transforms.RandomVerticalFlip())
    if rotaions:
        training_transform.append(MyRotationTransform([0, 90, 180, 270]))
    if augmentations == 'RandStainNA':
        training_transform = training_transform + [
            # torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            torchvision.transforms.ToPILImage(),
                                                   RandStainNA(yaml_file='RandStainNA/CRC_LAB_randomTrue_n0.yaml', std_hyper=-0.3, probability=1.0,distribution='normal', is_train=True),
                                                   torchvision.transforms.ToTensor(),
                                                   ]
    elif augmentations == 'AutoAugment':
        training_transform.append(torchvision.transforms.AutoAugment())
    elif augmentations == 'RandAugment':
        training_transform.append(torchvision.transforms.RandAugment())
    elif augmentations == 'AugMix':
        training_transform.append(torchvision.transforms.AugMix())
    training_transform.append(torchvision.transforms.ConvertImageDtype(torch.float32))
    if normalize:
        training_transform.append(torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return torchvision.transforms.Compose(training_transform)

def get_test_transform(normalize = True):
    test_transform = [torchvision.transforms.ConvertImageDtype(torch.float32)]
    if normalize:
        test_transform.append(torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return torchvision.transforms.Compose(test_transform)

class CycifImageDatasetFromCsv(torch.utils.data.Dataset):
    def __init__(self, df_file, img_dir_prefix = "", sample_cols = ['Sample','x','y'], 
                 label_col = ['cells', 'Ki67.mean', 'FOXP3.mean', 'PD1.mean', 'PDL1.mean'], 
                 sample_prefix = 'WD-76845-',
                 transform=None, target_transform=None):
        self.df = pd.read_csv(df_file)
        self.sample_cols = sample_cols
        self.label_col = label_col
        self.df = self.df[self.sample_cols + self.label_col]
        self.img_dir_prefix = img_dir_prefix
        self.sample_prefix = sample_prefix
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample, x, y = row[self.sample_cols]
        x, y = int(x), int(y)
        if self.sample_prefix == 'WD-76845-':
            sample = int(sample)
            sample_str = f'{self.sample_prefix}{sample:03}'
        else:
            sample_str = f'{self.sample_prefix}{sample}'
        img_path = os.path.join(self.img_dir_prefix, sample_str, f'{sample_str}_{x}_{y}.png')
        image = torchvision.io.read_image(img_path)
        labels = row[self.label_col].astype(float)
        labels = torch.tensor(labels.values, dtype = torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        # print('CycifImageDatasetFromCsv after', image.shape, type(image), image.dtype, labels.shape, type(labels), labels.dtype)
        return image, labels
    
class CycifImageTrainDatasetFromCsv(CycifImageDatasetFromCsv):
    def __init__(self, *args, **kwargs):
        kwargs['transform'] = get_training_transform(kwargs.get('normalize', True))
        super().__init__(*args, **kwargs)

class CycifImageTestDatasetFromCsv(CycifImageDatasetFromCsv):
    def __init__(self, *args, **kwargs):
        kwargs['transform'] = get_test_transform(kwargs.get('normalize', True))
        super().__init__(*args, **kwargs)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, 
                 shuffle_val_dataloader=False, shuffle_test_dataloader=True, common_args = {}, test_batch_size = None):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = batch_size if test_batch_size is None else test_batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.common_args = common_args
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = partial(self._train_dataloader, shuffle=shuffle_test_dataloader)
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        for k in self.dataset_configs:
            for k_tag in common_args:
                self.dataset_configs[k]['params'][k_tag] = common_args[k_tag]

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        self.dataloaders = {}
        if "train" in self.datasets:
            self.dataloaders["train"] = self.train_dataloader
        if "validation" in self.datasets:
            self.dataloaders["validation"] = self.val_dataloader
        if "test" in self.datasets:
            self.dataloaders["test"] = self.test_dataloader
        if "predict" in self.datasets:
            self.dataloaders["predict"] = self.predict_dataloader
        
    def _train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(self.datasets["train"], 
                                           batch_size=self.batch_size,
                                            num_workers=self.num_workers, 
                                            shuffle=shuffle)

    def _val_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(self.datasets["validation"],
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(self.datasets["test"],
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(self.datasets["predict"],
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)