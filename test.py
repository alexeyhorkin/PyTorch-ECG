import torch
import utils
import argparse
import models
import numpy as np
from dataset import get_dataset

def evaluate(model, test_dataloader, device, criterior):
    test_loss, accuracy, count = 0, 0, 0
    # treshold = args.trs_hold
    model.eval()
    with torch.no_grad():
        for batch_ind, (input, target) in enumerate(test_dataloader):
            input, target = input.to(device), target.to(device) 
            output = model(input)
            output_after_tresh = utils.apply_trashold(output, 0.1)
            tmp, length =  utils.get_accuracy(output_after_tresh, target)
            accuracy+=tmp
            count+=length
            test_loss+=criterior(output, target).item()
        print(f'Test loss is:{test_loss/len(test_dataloader):.5}')
        print(f'Accuracy is: {accuracy/count:.5}')

def foo(index_ones, index_max):
    arr = []
    for i in index_ones:
        if i in index_max:
            arr.append(True)
        else:
            arr.append(False)
    arr = np.array(arr)
    if arr.astype(np.float32).sum()==len(arr):
        return arr, True
    else:
        return arr, False

def test(args, model, device):
    torch.manual_seed(12)
    train_dataset, test_dataset = get_dataset(args)
    for batch_ind, (input, target) in enumerate(test_dataset):
        input, target = input.to(device), target.to(device)
        output = model(input)
        index_ones = np.argwhere(target.view(-1).cpu().numpy()==1).reshape(np.count_nonzero(target.cpu().numpy()))
        item_max, index_max = torch.topk(output.view(-1), len(index_ones))
        arr, is_true = foo(index_ones, index_max.cpu().numpy())
        # print(index_max)
        # print(index_ones)
        # print(item_max)
        print(arr, is_true)
        print()
        

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=40, help='Size of batch during training')
    parser.add_argument('--test_batch_size', type=int, default=12, help='Size of batch during testing')
    parser.add_argument('--num_workers', type=int, default=2, help='Count workers for data loading')
    parser.add_argument('--snapshot_path', type=str, default='ecg_data_200.json', help='Path to data file')
    parser.add_argument('--path_to_DataFile', type=str, default='ecg_data_200.json', help='Path to data file')
    parser.add_argument('--is_shuffle', type=bool, default=True, help='Shuffle train or not')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.snapshot_path).to(device)
    test(args, model, device)