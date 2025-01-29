import os
import copy

import pandas as pd

pd.options.display.float_format = '{:.2f}'.format
from utils import preety_print_model_ratings, get_rank, preprocess_data, compute_mle_elo_dict

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--rigging_mode', type=str, default='omni_bt_diff')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--classifier_acc', type=float, default=1.0)
parser.add_argument('--model_name_list', nargs='+', default=['phi-3-mini-4k-instruct-june-2024'])

args = parser.parse_args()


X_initial, Y_initial, win_matrix_initial, sample_weights_ori = preprocess_data('data/data_x.npy', 'data/data_y.npy','data/vh_win_matrix.csv')
model_name_sorted = []
for model_name in win_matrix_initial.index:
    model_name_sorted.append(model_name) if model_name not in model_name_sorted else None


print('Calculate Initial Rating')
elo_ratings, _ = compute_mle_elo_dict([], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=sample_weights_ori)
initial_ranking = preety_print_model_ratings(elo_ratings)

print('---------------initial ranking---------------')
print(initial_ranking)
print('---------------------------------------------')


result_dict = {}

for target_model in args.model_name_list:
    print(target_model)
    
    sample_weights_tmp = copy.deepcopy(sample_weights_ori)
    ori_rank = get_rank(initial_ranking, target_model)
    
    
    
    with open(f'voting_output/{target_model}_{args.rigging_mode}_acc_{args.classifier_acc}_prob_dec_{args.beta}.json') as f:
        manipulated_battle_dict = json.load(f)
    
   

    with open(f'data/vo_1.7m.json') as f:
        normal_battles_dict = json.load(f)
    
    manipulated_battle_list = []
    for idx, key_idx in enumerate(manipulated_battle_dict.keys()):
        manipulated_battle_list.append(manipulated_battle_dict[key_idx])
    
    final_ranking, sample_weights_tmp = compute_mle_elo_dict(manipulated_battle_list, X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=sample_weights_tmp)
    final_ranking = preety_print_model_ratings(final_ranking)
    final_rank = get_rank(final_ranking, target_model)
   
    result_dict[f'{target_model}_0'] = {'ori_rank': ori_rank, 'final_rank': final_rank}
   
    for vote_num in [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]:
        current_battle_list = []
        for idx, key_idx in enumerate(normal_battles_dict.keys()):
            if idx < vote_num - 10000:
                continue
            if idx == vote_num:
                break
            current_battle_list.append(normal_battles_dict[key_idx])
        assert len(current_battle_list) == 10000
       
        final_ranking, sample_weights_tmp = compute_mle_elo_dict(current_battle_list, X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=sample_weights_tmp)
        final_ranking = preety_print_model_ratings(final_ranking)

        final_rank = get_rank(final_ranking, target_model)
        result_dict[f'{target_model}_{int(vote_num)}'] = {'ori_rank': ori_rank, 'final_rank': final_rank}
    
        os.makedirs('voting_output/rigging_vo/', exist_ok=True)
        with open(f'voting_output/rigging_vo/{args.rigging_mode}.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

    