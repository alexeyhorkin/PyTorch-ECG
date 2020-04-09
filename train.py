import torch
import argparse
import models
import test
import torch.nn as nn
import numpy as np
import os
from dataset import get_dataset
import matplotlib.pyplot as plt 

# def train(model, device, args, optimazer, criterior, train_dataset, test_dataset):
#     torch.manual_seed(12)
#     for i in range(args.eph):
#         train_loss = 0
#         for batch_ind, (input, target) in enumerate(train_dataset):
#             optimazer.zero_grad()
#             input, target = input.to(device), target.to(device)
#             output = model(input)
#             loss = criterior(output, target)
#             loss.backward()
#             optimazer.step()
#             train_loss+=loss.item()

#         print(f'Epoch {i+1}/{args.eph}')
#         test.evaluate(model, test_dataset, device, criterior)
#         print(f'Train loss is: {train_loss/len(train_dataset)} \n')
#         # save model
#         if i%30==0:
#             if not os.path.exists(args.save_path):
#                 os.mkdir(args.save_path)
#             torch.save(model, os.path.join(args.save_path, 'model_'+str(i)+'_epoch.pth'))
        


def main(args, model):
    torch.manual_seed(12)
    train_dataset, test_dataset = get_dataset(args)

    a,b = next(iter(train_dataset))
    print(b.size())
    for i in range(args.train_batch_size):
        x, y = a[i], b[i]
        plt.plot(x.numpy())
        plt.scatter(y.numpy(), [0]*len(y.numpy()), c='r')
        plt.show()
        # if y.numpy().max() > 550 or y.numpy().min() < 0:
            # print('index is !!!!!', i) 

    print("TEST!!!")
    a,b = next(iter(test_dataset))
    print(b.size())
    for i in range(args.test_batch_size):
        x, y = a[i], b[i]
        plt.plot(x.numpy())
        plt.scatter(y.numpy(), [0]*len(y.numpy()), c='r')
        plt.show()
        # if y.numpy().max() > 550 or y.numpy().min() < 0:
            # print('index is !!!!!', i) 

    # optimazer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterior = nn.MSELoss()
    # criterior = nn.BCELoss()
    # train(model, device, args, optimazer, criterior, train_dataset, test_dataset)



if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=160, help='Size of batch during training')
    parser.add_argument('--test_batch_size', type=int, default=40, help='Size of batch during testing')
    parser.add_argument('--num_workers', type=int, default=0, help='Count workers for data loading')
    parser.add_argument('--path_to_DataFile', type=str, default='fix_data.json', help='Path to data file')
    parser.add_argument('--save_path', type=str, default='Saved_models', help='Path to save models during training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eph', type=int, default=1, help='Count of epoches')
    parser.add_argument('--cycle_length', type=int, default=250, help='Size of data = 2*cycle_length')
    parser.add_argument('--is_shuffle', type=bool, default=False, help='Shuffle test and train or not')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.CNN().to(device)
    main(args, model)

