import os
import torch
import utils
import argparse
import models
import numpy as np
import math as mt
from dataset import get_dataset
from utils import visualize_out, get_transforms, plot_learning

def evaluate(model, test_dataloader, device, criterior):
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_ind, (inputs, target) in enumerate(test_dataloader):
            inputs, target = inputs.to(device), target.to(device) 
            outputs = model(inputs)
            test_loss+=criterior(outputs, target).item()
    
    return test_loss


def test(args, model, device):
    torch.manual_seed(12)

    train_transform, test_transform = get_transforms()
    _, test_dataset = get_dataset(args,train_transform, test_transform)
    criterior = torch.nn.MSELoss()
    test_loss = evaluate(model, test_dataset, device, criterior)
    print(f'Test loss is {test_loss/len(test_dataset):.5f}')
    visualize_out(model, test_dataset, device)

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=40, help='Size of batch during training')
    parser.add_argument('--test_batch_size', type=int, default=10, help='Size of batch during testing')
    parser.add_argument('--is_shuffle', type=bool, default=True, help='Shuffle train or not')
    parser.add_argument('--num_workers', type=int, default=2, help='Count workers for data loading')
    parser.add_argument('--snapshot_path', type=str, default='lol.pth', help='Path to data file')
    parser.add_argument('--path_to_DataFile', type=str, default='fix_data.json', help='Path to data file')
    parser.add_argument('--cycle_length', type=int, default=256, help='Size of data = 2*cycle_length')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.snapshot_path).to(device)
    test(args, model, device)