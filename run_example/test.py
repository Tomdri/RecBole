# @Time   : 2021/03/20
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn


"""
Case study example
===================
Here is the sample code for the case study in RecBole.
"""


import numpy as np
import os
import torch
from recbole.data.interaction import Interaction
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model
from recbole.model.sequential_recommender.fpmc import FPMC
import pandas as pd
import matplotlib.pyplot as plt
import onnx
from onnx import numpy_helper

if __name__ == '__main__':
    data = pd.read_csv('./run_example/items.csv', header=None, names=['order_id', 'item_id', 'item_name', 'timestamp'])

    with torch.no_grad():
        try:
            config, model, dataset, train_data, valid_data, test_data1 = load_data_and_model(
                model_file='./saved/code_models/NARMcode3iter.pth'
            )
        except RuntimeError as e:
            print('Error', e)

        model.eval()

        seq_input_inter = Interaction({
            'product_code_list': torch.LongTensor([[1,2]]),
            'item_length': torch.LongTensor([2]),
        })
        seq_input_inter.to(config.device)

        try:
            scores = model.full_sort_predict(seq_input_inter)
        except Exception as e:
            print(e)

        top_items_internal_ids = scores[0].numpy().argsort()[::-1][:15]
        top_items_internal_ids_list = [i for i in top_items_internal_ids]
        print(top_items_internal_ids_list)
        top_items_ids = dataset.id2token(dataset.iid_field, top_items_internal_ids_list)
        print(top_items_ids)