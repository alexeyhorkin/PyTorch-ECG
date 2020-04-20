import os
import torch
import json
from torch.utils.data import  DataLoader
from utils import ECG_dataset, Peaks_dataset


def get_dataset(args, train_transform, test_transform):
    ''' 
    Special func for return train and test dataloader 
    '''

    train_ecg_dataset = ECG_dataset(args.path_to_DataFile, args.cycle_length, 'qrs',  is_train=True, transform=train_transform)
    test_ecg_dataset =  ECG_dataset(args.path_to_DataFile, args.cycle_length, 'qrs',  is_train=False, transform=test_transform)

    train_dataloader = DataLoader(dataset = train_ecg_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=args.is_shuffle)
    test_dataloader = DataLoader(dataset = test_ecg_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataloader, test_dataloader

def get_dataset_peaks(args, train_transform, test_transform):

    train_ecg_dataset = Peaks_dataset(args.cycle_length,  is_train=True, transform=train_transform)
    test_ecg_dataset =  Peaks_dataset(args.cycle_length,  is_train=False, transform=test_transform)

    train_dataloader = DataLoader(dataset = train_ecg_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=args.is_shuffle)
    test_dataloader = DataLoader(dataset = test_ecg_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataloader, test_dataloader    
