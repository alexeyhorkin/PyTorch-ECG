import os
import torch
import json
import pywt
import pickle
import numpy as np
import random as rd
import math as mt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from numba import jit
from statsmodels.robust import mad

class ECG_dataset(Dataset):
    def __init__(self, path_to_DataFile, cycle_lenght, type_of_wave, is_train=True, transform=None):
        ''' 
        Class for custom dataset for ecg signals
        '''
        self.data_dict = json.load(open(os.path.join(path_to_DataFile), 'r') )
        self.path_to_DataFile = path_to_DataFile
        self.transform = transform
        self.otvedenie = 'i'
        self.keys, self.procent_of_train = list(self.data_dict.keys()), 0.8
        self.count_data = len(self.keys)
        rd.seed(10) # set a seed for taking random peaks
        self.cycle_lenght = cycle_lenght
        self.type_of_wave = type_of_wave
        self.is_train = is_train

        if self.is_train:
            self.len = int(self.count_data*self.procent_of_train)
            self.keys_for_get_item = self.keys
        else:
            self.len = int(self.count_data*(1-self.procent_of_train))+1
            self.keys_for_get_item = self.keys[self.count_data-1:self.count_data-self.len-1:-1]

    def __len__(self):
            return self.len
    
    def __getitem__(self, index):
        if self.is_train:
            if self.type_of_wave == 'p':
                if index in [11, 12, 29, 40, 60, 61, 63, 76, 88, 89, 94, 95, 96, 108, 112, 141, 154]:
                    index = 1
        else:
            if self.type_of_wave == 'p':
                if index in [1, 27, 30, 39]:
                    index-=1

        leads = self.data_dict[self.keys_for_get_item[index]]['Leads']
        signal = leads[self.otvedenie]["Signal"]
        labels = [ (tmp['Index'],tmp['Name'])  for tmp in leads[self.otvedenie]['Morphology'][self.type_of_wave] if tmp['Name'] not in ['xtd_point', 'q', 's'] ]
        peaks_idx = [ tmp['Index']  for tmp in leads[self.otvedenie]['Morphology']['qrs'] if tmp['Name']=='r' ]
        peak = rd.sample(peaks_idx, 1)[0] # take a random peak
        label = get_label(peak, labels, self.cycle_lenght, self.type_of_wave)

        lol = 0
        while (peak-self.cycle_lenght<0) or (peak+self.cycle_lenght>len(signal)) or label is None: # resave the peak if index out of range or label None
            peak = rd.sample(peaks_idx, 1)[0] # take a random peak
            label = get_label(peak, labels, self.cycle_lenght, self.type_of_wave)
            lol+=1
            if lol==30:
                print('break(((')
                break

        # print('peak is ', peak)
        step = int(30*rd.random())
        shift = 30 - step
        if self.is_train:
            res = signal[peak-self.cycle_lenght- shift:peak+self.cycle_lenght+ 80 + step]
            label+=shift
        else:
            res = signal[peak-self.cycle_lenght-80 -step:peak+self.cycle_lenght + shift]
            label +=step + 80
        sample = (np.array(res), label)
        if self.transform:
            sample = self.transform(sample)
        
        return sample 



def get_label(peak, labels, lenght, type_wave):
    indexes_labels = np.array([i[0] for i in labels])
    idx = (np.abs(indexes_labels-peak)).argmin()
    if type_wave =='t':
        if peak>labels[idx][0] or idx+2>len(labels)-1 or labels[idx][1]!='t_onset':
            return None
        return np.array([labels[idx][0] - peak, labels[idx+1][0]-peak, labels[idx+2][0] - peak])+lenght
    if type_wave =='p':
        if peak<labels[idx][0] or idx-2<0 or labels[idx][1]!='p_offset':
            return None
        return np.array([labels[idx][0] - peak, labels[idx-1][0]-peak, labels[idx-2][0] - peak])+lenght
    if type_wave =='qrs':
        if labels[idx][1]!='r':
            return None
        return np.array([labels[idx-1][0] - peak, labels[idx][0]-peak, labels[idx+1][0] - peak])+lenght    

def find_peaks_div(x, scope_max=10, scope_null=50):
    lenght = x.shape[0]
    y0 = np.zeros(lenght)
    y1 = np.zeros(lenght)
    y2 = np.zeros(lenght)
    y3 = np.zeros(lenght)

    for i in range(lenght-2):
        y0[i+2] = mt.fabs(x[i+2]-x[i])
    for i in range(lenght-4):
        y1[i+4] = mt.fabs(x[i+4]-2*x[i+2]+ x[i])
    for i in range(lenght-4):
        y2[i+4] = 1.3*y0[i+4]+1.1*y1[i+4]
    for i in range(lenght-4-7):
        for k in range(7):
            y3[i] += y2[i+4-k]
        y3[i] /= 8
    max_idx = []
    curr_max = max(y3)
    curr_argmax = np.argmax(y3)
    true_argmax = np.argmax(x[max(0,curr_argmax-scope_max):min(curr_argmax+scope_max,lenght)])

    max_idx.append(max(0, curr_argmax-scope_max) + true_argmax)
    y3[max(0,curr_argmax-scope_null):min(curr_argmax+scope_null,lenght)] *= 0

    prev_max = curr_max
    curr_max = max(y3)

    while (prev_max - curr_max) < (prev_max / 4.0):
        curr_argmax = np.argmax(y3)
        true_argmax = np.argmax(x[max(0,curr_argmax-scope_max):min(curr_argmax+scope_max,lenght)])
        max_idx.append(max(0, curr_argmax-scope_max) + true_argmax)
        y3[max(0,curr_argmax-scope_null):min(curr_argmax+scope_null,lenght)] *= 0
        prev_max = curr_max
        curr_max = max(y3)
    return max_idx

def wavelet_smooth(X, wavelet="db4", level=1, title=None):
    coeff = pywt.wavedec(X, wavelet, mode="per")
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(X)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    y = pywt.waverec(coeff, wavelet, mode="per")
    return y


def visualize_out(model, test_dataloader, train_dataloader,  device):
    inputs, targets = next(iter(train_dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    inputs, targets, outputs = inputs.cpu().numpy(), targets.cpu().numpy(), outputs.cpu().numpy()
    inputs = inputs.reshape(inputs.shape[0],-1)
    size, indx = len(targets[0]), 0
    count = int(mt.sqrt(targets.shape[0])) 
    fig, ax = plt.subplots(count, count)
    for i in range(count):
        for j in range(count):
            if indx < targets.shape[0] and i*count+j < count**2:
                ax[i][j].plot(inputs[indx])
                ax[i][j].scatter(targets[indx], [0]*size, c='g')
                ax[i][j].scatter(outputs[indx], [0]*size, c='r')
                indx+=1

    plt.show()

    inputs, targets = next(iter(test_dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    inputs, targets, outputs = inputs.cpu().numpy(), targets.cpu().numpy(), outputs.cpu().numpy()
    inputs = inputs.reshape(inputs.shape[0],-1)
    size, indx = len(targets[0]), 0
    count = int(mt.sqrt(targets.shape[0])) 
    fig, ax = plt.subplots(count, count)
    for i in range(count):
        for j in range(count):
            if indx < targets.shape[0] and i*count+j < count**2:
                ax[i][j].plot(inputs[indx])
                ax[i][j].scatter(targets[indx], [0]*size, c='g')
                ax[i][j].scatter(outputs[indx], [0]*size, c='r')
                indx+=1

    plt.show()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        array, target = sample
        array, target = np.array(array[None,:]), np.array([target]) 
        return (torch.from_numpy(array).float(), torch.from_numpy(target).float()) #[batch_size, 1, size_of_data] and [batch_size, size_of_data] 

class ToChoise(object):
    def __init__(self, index=0):
        self.index = index
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        array, target = sample
        array, target = np.array(array[None,:]), np.array([target[self.index]])
        return (torch.from_numpy(array).float(), torch.from_numpy(target).float()) #[batch_size, 1, size_of_data] and [batch_size, size_of_data]

def get_transforms():
    return (ToChoise(1), ToChoise(1)) 

def get_transforms_for_dataset_peaks():
    return (ToTensor(), ToTensor()) 


def plot_learning(path_to_data):
    ms=3
    with open(path_to_data, 'rb') as f:
        data_dict = pickle.load(f)
        keys = list(data_dict.keys())
    plt.title('Learning graphs')
    plt.subplot(2,1,1)

    plt.plot(range(1,len(data_dict[keys[1]])+1), data_dict[keys[1]], marker='o', ms=ms)
    plt.plot(range(1,len(data_dict[keys[1]])+1), data_dict[keys[2]], marker='o', ms=ms)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend(['Train loss', 'Test loss'], loc='upper left')
    plt.subplot(2,1,2)
    plt.plot(range(1,len(data_dict[keys[0]])+1)[::6], data_dict[keys[0]][::6], marker='o', ms=ms)
    plt.xlabel('number of mini-batch')
    plt.ylabel('loss')
    plt.legend(['Train loss on batch'], loc='upper left')
    plt.show()

class Peaks_dataset(Dataset):
    def __init__(self, size, is_train=True, transform=None):
        ''' 
        Class for custom dataset for ecg signals
        '''
        self.transform = transform
        self.procent_of_train = 0.8
        rd.seed(10) # set a seed for taking random peaks
        self.size = size
        self.is_train = is_train
        self.amplitude = 100.0
        if self.is_train: self.len = 160
        else: self.len = 40


    def __len__(self):
            return self.len
    
    def __getitem__(self, index):
        arr = np.zeros(self.size)
        step = int(120*rd.random())
        index_middle = (self.size-1)//2
        if self.is_train:
            index_peak = index_middle - 10 - step
        else:
            index_peak = index_middle + 10 + step
        arr[index_peak]=self.amplitude
        sample = (arr ,index_peak)
        if self.transform:
            sample = self.transform(sample)
        
        return sample 