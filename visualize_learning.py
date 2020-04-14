import os
import utils
import argparse
import pickle
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.subplots import make_subplots


def get_data():
    data = {}
    paths = [ tmp for tmp in os.listdir(path='./') if 'exp' in tmp.lower() ]

    for path in paths:
        path_to_pickle = os.path.join(path, 'metrics', 'metrics_learning.pickle')
        with open( path_to_pickle, 'rb') as curr_f:
            curr_data = pickle.load(curr_f)
        data[path] = curr_data

    return data, list(data.keys())

def plot(data, *names):
    print('Including following paths:')
    print(names)

    fig = make_subplots(rows=2, cols=1,  subplot_titles=("Losses during train", "Batch loss on train"))
    for tmp in names:
        keys = list(data[tmp].keys())
        size = len(data[tmp][keys[1]])
        size2 = len(data[tmp][keys[2]]) 
        fig.add_trace(go.Scatter(
            x=np.arange(1,size),
            y=data[tmp][keys[1]],
            mode='lines+markers',
            name=tmp+' Train loss'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(1,size),
            y=data[tmp][keys[2]],
            mode='lines+markers',
            name=tmp+' Test loss'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(1,size2),
            y=data[tmp][keys[0]],
            mode='lines+markers',
            name=tmp+' Train loss on batch'
        ), row=2, col=1)

    fig.update_xaxes(title_text="Epoches", row=1, col=1)
    fig.update_xaxes(title_text="Batch numbers", row=2, col=1)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    return fig


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path_', type=str, default='Current_exp', help='Path to save models during training')
    args = parser.parse_args()
    
    data, exp_names = get_data()
    plot(data, *exp_names).show()