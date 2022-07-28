# script that converts a model to a .onnx file

import torch.onnx
import torch
from recbole.model.sequential_recommender.narm import NARM
from recbole.data.interaction import Interaction
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import load_data_and_model
import pandas as pd
import numpy as np
import pickle
import json

if __name__ == "__main__":
    # config, model, dataset, train_data, valid_data, test_data1 = load_data_and_model(
    #     model_file='./saved/code_models/NARMcode3iter.pth'
    # )
    # model.eval()

    # dummy_input = (torch.LongTensor([[0]*2275]), torch.LongTensor([2275]))

    # torch.onnx.export(model, dummy_input, f='./run_example/NARM.onnx',
    #  verbose=True, input_names=['products_seq', 'products_seq_length'],
    #   output_names=['scores'], opset_version=11, dynamic_axes={'products_seq': {1: 'products_sequence'}})

    data = pd.read_csv('./run_example/items.csv', header=None, names=['order_id', 'item_id', 'item_name', 'timestamp'])

    external_to_internal = {}
    internal_to_external = {}
    
    data = data.loc[data.groupby('order_id').item_name.transform(len) > 1]

    data = data['item_name'].unique()

    for i in range(len(data)):
        external_to_internal[data[i]] = i+1
        internal_to_external[i+1] = data[i]

    with open('./run_example/external_to_internal.csv', 'w') as f:
        for key, value in external_to_internal.items():
            f.write("%s,%s\n" % (key, value))

    with open('./run_example/internal_to_external.csv', 'w') as f:
        for key, value in internal_to_external.items():
            f.write("%s,%s\n" % (key, value))