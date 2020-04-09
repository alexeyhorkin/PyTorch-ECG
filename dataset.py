import os
import torch
import json
from torch.utils.data import  DataLoader
from utils import ECG_dataset


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        array, target = sample
        return (torch.from_numpy(array).float(), torch.from_numpy(target).float())

class ToChoise(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample,index):
        array, target = sample
        target = target[index]
        return (torch.from_numpy(array).float(), torch.from_numpy(target).float())


def get_dataset(args):
    ''' 
    Special func for return train and test dataloader 
    '''

    train_ecg_dataset = ECG_dataset(args.path_to_DataFile, args.cycle_length, 't',  is_train=True, transform=ToTensor())
    test_ecg_dataset =  ECG_dataset(args.path_to_DataFile, args.cycle_length, 't',  is_train=False, transform=ToTensor())

    train_dataloader = DataLoader(dataset = train_ecg_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=args.is_shuffle)
    test_dataloader = DataLoader(dataset = test_ecg_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataloader, test_dataloader
