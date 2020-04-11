import os
import torch
import argparse
import models
import test
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np
from dataset import get_dataset
from utils import get_transforms

def train(model, device, args, optimazer, criterior, train_dataset, test_dataset):
    torch.manual_seed(12)
    metrics = {'Global_train_loss': [], 'Epoches_train_loss':[], 'Epoches_test_loss':[] }
    for i in range(args.eph):
        train_loss = 0
        for batch_ind, (inputs, target) in enumerate(train_dataset):
            optimazer.zero_grad()
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            loss = criterior(output, target)
            loss.backward()
            optimazer.step()
            train_loss+=loss.item()
            metrics['Global_train_loss'].append(loss.item())

        test_loss = test.evaluate(model, test_dataset, device, criterior)
        print(f'Epoch {i+1}/{args.eph}')
        print(f'Train loss is: {train_loss/len(train_dataset):.5f}')
        print(f'Test loss is: {test_loss/len(test_dataset):.5f} \n')
        metrics['Epoches_train_loss'].append(train_loss/len(train_dataset))
        metrics['Epoches_test_loss'].append(test_loss/len(test_dataset))
        # save model
        if i%args.save_every==0:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            torch.save(model, os.path.join(args.save_path, 'model_'+str(i)+'_epoch.pth'))
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'metrics')):
        os.mkdir(os.path.join(args.save_path, 'metrics'))

    with open( os.path.join(args.save_path, 'metrics', 'metrics_learning.pickle'), 'wb') as f:
        pickle.dump(metrics, f)
        


def main(args, model, device):
    torch.manual_seed(12)
    train_transform, test_transform = get_transforms()
    train_dataset, test_dataset = get_dataset(args, train_transform, test_transform)
    optimazer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterior = nn.MSELoss()
    train(model, device, args, optimazer, criterior, train_dataset, test_dataset)



if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=20, help='Size of batch during training')
    parser.add_argument('--test_batch_size', type=int, default=40, help='Size of batch during testing')
    parser.add_argument('--num_workers', type=int, default=0, help='Count workers for data loading')
    parser.add_argument('--is_shuffle', type=bool, default=False, help='Shuffle train or not')
    parser.add_argument('--path_to_DataFile', type=str, default='fix_data.json', help='Path to data file')
    parser.add_argument('--save_path', type=str, default='Current_exp', help='Path to save models during training')
    parser.add_argument('--save_every', type=int, default=5, help='Save every # epoches')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eph', type=int, default=2, help='Count of epoches')
    parser.add_argument('--cycle_length', type=int, default=250, help='Size of data = 2*cycle_length')
    parser.add_argument('--md_type', type=str, choices=['MLP'], required=True, help='Name of model to train')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, args.md_type)(input_size=args.cycle_length*2, output_size=1).to(device)
    print(model)
    main(args, model, device)

