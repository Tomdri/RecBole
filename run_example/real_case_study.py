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

if __name__ == '__main__':
    data = pd.read_csv('./run_example/items.csv', header=None, names=['order_id', 'item_id', 'item_name', 'timestamp'])
    model_list = os.listdir('saved/code_models/')

    with torch.no_grad():
        test_data = data.groupby('order_id').apply(lambda x: x.item_name.tolist())
        test_data = test_data[test_data.apply(lambda x: len(x) >= 20)]
        test_data = test_data.sample(1)
        test_data = test_data.to_dict()
        results = {}

        for model_name in model_list:
            try:
                config, model, dataset, train_data, valid_data, test_data1 = load_data_and_model(
                    model_file='./saved/code_models/' + model_name
                )
            except RuntimeError as e:
                print('Error', e)

            model.eval()

            model_result = []

            for i in range(1, len(test_data[list(test_data.keys())[0]])):
                sequence_data = {k: v[:i] for k, v in test_data.items()}
                prediction_data = {k: v[i:] for k, v in test_data.items()}
                for order_id, item_ids in sequence_data.items():
                    item_ids = [i for i in item_ids]
                    items_internal_ids = dataset.token2id(dataset.iid_field, [item_ids])
                    seq_input_inter = Interaction({
                        'product_code_list': torch.LongTensor([i for i in items_internal_ids]),
                        'item_length': torch.LongTensor([len(items_internal_ids[0])]),
                    })
                    seq_input_inter.to(config.device)

                    try:
                        scores = model.full_sort_predict(seq_input_inter)
                    except Exception as e:
                        print(e)
                        continue

                    top_items_internal_ids = scores[0].numpy().argsort()[::-1][:len(prediction_data[order_id])]
                    top_items_internal_ids_list = [i for i in top_items_internal_ids]
                    top_items_ids = dataset.id2token(dataset.iid_field, top_items_internal_ids_list)

                    items_predicted = 0
                    for j in top_items_ids:
                        if j in prediction_data[order_id]:
                            items_predicted += 1
                    model_result.append(round(((items_predicted / len(prediction_data[order_id])) * 100), 2))
            results[model_name.split('-')[0]] = model_result

        for result in results:
            print(result, results[result])
            print('Avg:', round(np.mean(results[result]), 2))
            plt.plot(results[result])
        
        plt.legend([result for result in results])
        plt.show()