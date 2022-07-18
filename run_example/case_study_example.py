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

if __name__ == '__main__':
    data = pd.read_csv('./run_example/items.csv', header=None, names=['order_id', 'item_id', 'item_name', 'timestamp'])
    model_list = os.listdir('saved/sequential/')
    with torch.no_grad():
        test_data = data.groupby('order_id').apply(lambda x: x.item_id.tolist())
        test_data = test_data[test_data.apply(lambda x: len(x) >= 20)]
        test_data = test_data.sample(100)
        test_data = test_data.to_dict()
        sequence_data = {k: v[:10] for k, v in test_data.items()}
        prediction_data = {k: v[10:] for k, v in test_data.items()}
        results = {}

        for model_name in model_list:
            if model_name == 'xSINE-Jul-14-2022_17-11-49.pth' or model_name == 'xSTAMP-Jul-14-2022_19-31-25.pth':
                continue
            try:
                config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
                    model_file='./saved/sequential/' + model_name
                )
            except RuntimeError as e:
                print('Error', e)
                continue

            model.eval()

            model_result = []
            for order_id, item_ids in sequence_data.items():
                item_ids = [str(i) for i in item_ids]
                items_internal_ids = dataset.token2id(dataset.iid_field, [item_ids])
                order_internal_id = dataset.token2id(dataset.uid_field, [str(order_id)])
                seq_input_inter = Interaction({
                    # 'user_id': torch.LongTensor(order_internal_id),
                    'item_id_list': torch.LongTensor([i for i in items_internal_ids]),
                    'item_length': torch.LongTensor([len(items_internal_ids[0])]),
                })
                seq_input_inter.to(config.device)

                try:
                    scores = model.full_sort_predict(seq_input_inter)
                except Exception as e:
                    print(e)
                    continue

                top_items_internal_ids = scores.numpy().argsort()[::-1][:len(prediction_data[order_id])] if model_name[0] != 'x' else scores[0].numpy().argsort()[::-1][:len(prediction_data[order_id])]
                top_items_internal_ids_list = [i for i in top_items_internal_ids]
                top_items_ids = dataset.id2token(dataset.iid_field, top_items_internal_ids_list)

                items_predicted = 0
                for j in top_items_ids:
                    if int(j) in prediction_data[order_id]:
                        items_predicted += 1
                model_result.append(round(((items_predicted / len(prediction_data[order_id])) * 100), 2))
            results[model_name.split('-')[0]] = model_result

        for result in results:
            print(result, results[result])
            print('Average', np.mean(results[result]))

    #     score_results = [0]*100
    #     model_results = ['']*100

    # for i in range(100):
    #     for model_name in model_list:
    #         model_name = model_name.split('-')[0]
    #         if results[model_name][i] > score_results[i]:
    #             score_results[i] = results[model_name][i]
    #             model_results[i] = model_name
    #         elif results[model_name][i] == score_results[i]:
    #             model_results[i] = model_name if model_results[i] == '' else (model_results[i] + ' ' + model_name)
    # print(model_results)

    # models_scores = {}
    # BPR = 0
    # DMF = 0
    # FISM = 0
    # LightGCN = 0
    # LINE = 0
    # MacridVAE = 0
    # SpectralCF = 0
    # for best_model in model_results:
    #     if 'BPR' in best_model:
    #         BPR += 1
    #     elif 'DMF' in best_model:
    #         DMF += 1
    #     elif 'FISM' in best_model:
    #         FISM += 1
    #     elif 'LightGCN' in best_model:
    #         LightGCN += 1
    #     elif 'LINE' in best_model:
    #         LINE += 1
    #     elif 'MacridVAE' in best_model:
    #         MacridVAE += 1
    #     elif 'SpectralCF' in best_model:
    #         SpectralCF += 1

    # print('BPR', BPR)
    # print('DMF', DMF)
    # print('FISM', FISM)
    # print('LightGCN', LightGCN)
    # print('LINE', LINE)
    # print('MacridVAE', MacridVAE)
    # print('SpectralCF', SpectralCF)