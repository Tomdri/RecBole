from recbole.quick_start.quick_start import load_data_and_model
from recbole.model.sequential_recommender.fpmc import full_sort_topk

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='./saved/FPMC-Jul-05-2022_09-37-28.pth')

uid_series = dataset.token2id(dataset.uid_field, ['96047'])

topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
print(topk_score)  # scores of top 10 items
print(topk_iid_list)  # internal id of top 10 items
external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
print(external_item_list)  # external tokens of top 10 items