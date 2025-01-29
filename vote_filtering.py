import os
import copy
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format

from utils import preety_print_model_ratings, get_rank, preprocess_data, compute_mle_elo_dict

import argparse
import cudf
import json

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='omni_bt_diff')
parser.add_argument('--beta', type=float, default=1.0)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--classifier_acc', type=float, default=1.0)
parser.add_argument('--filter_threshold', type=float, default=0.7)
parser.add_argument('--model_name_list', nargs='+', default=['phi-3-mini-4k-instruct-june-2024'])



args = parser.parse_args()

K = 4
BASE = 10
SCALE = 400


# Initialize the rigging environment
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


initial_rating = {}
for idx, key in enumerate(initial_ranking['Model'].keys()):
    initial_rating[initial_ranking.loc[key, 'Model']] = initial_ranking.loc[key, 'Elo rating']        


result_dict = {}


for target_model in args.model_name_list:
   
    ori_rank = get_rank(initial_ranking, target_model)
    tot_dict = {}
    tot_battle_list = []

    with open(f'voting_output/{target_model}_{args.rigging_mode}_acc_{args.classifier_acc}_prob_dec_{args.beta}.json') as f:
        manipulated_battle_dict = json.load(f)
    
    for idx, key in enumerate(manipulated_battle_dict.keys()):
        model_a = manipulated_battle_dict[key]['model_a']
        model_b = manipulated_battle_dict[key]['model_b']
        winner = manipulated_battle_dict[key]['winner']
        
        ra = initial_rating[model_a]
        rb = initial_rating[model_b]
        
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        
        if (winner == 'model_b' and ea > args.filter_threshold) or (winner == 'model_a' and eb > args.filter_threshold):
         
            continue
            
        
        tot_dict[idx] = {'model_a': model_a, 'model_b': model_b, 'winner': winner}
       
    for idx, key_idx in enumerate(tot_dict.keys()):
        tot_battle_list.append(tot_dict[key_idx])

    final_ranking, _ = compute_mle_elo_dict(tot_battle_list, X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_ori))
    final_ranking = preety_print_model_ratings(final_ranking)
    final_rank = get_rank(final_ranking, target_model)
    
    result_dict[f'{target_model}'] = {'ori_rank': ori_rank, 'final_rank': final_rank}

    os.makedirs('voting_output/filtered_battles', exist_ok=True)
    with open(f'voting_output/filtered_battles/{args.mode}_thre_{args.filter_threshold}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)