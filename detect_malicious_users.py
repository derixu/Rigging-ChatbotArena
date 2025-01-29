import os

import math
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.2f}'.format
from utils import preety_print_model_ratings, preprocess_data, compute_mle_elo_dict

from tqdm import tqdm

import argparse
import json

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--rigging_mode', type=str, default='diff_greedy')
parser.add_argument('--users_num', type=int, default=100)
parser.add_argument('--vote_num_per_user', type=int, default=100)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--classifier_acc', type=float, default=1.0)
parser.add_argument('--random_round', type=int, default=100)
parser.add_argument('--model_name_list', nargs='+', default=['phi-3-mini-4k-instruct-june-2024'])



args = parser.parse_args()


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


win_dict = {}
for model in model_name_sorted:
    win_dict[model] = {}

for model_a in model_name_sorted:
    for model_b in model_name_sorted:
        if model_a != model_b:
            win_dict[model_a][model_b] = {'win': 0, 'lose': 0, 'tie': 0}


with open(f'data/vh.json') as f:
    vh_dict = json.load(f)
print(len(vh_dict.keys()))
for idx, key_idx in tqdm(enumerate(vh_dict)):
    model_a = vh_dict[key_idx]['model_a']
    model_b = vh_dict[key_idx]['model_b']
    winner = vh_dict[key_idx]['winner']
    if winner == 'model_a':
        win_dict[model_a][model_b]['win'] += 1
        win_dict[model_b][model_a]['lose'] += 1
    elif winner == 'model_b':
        win_dict[model_a][model_b]['lose'] += 1
        win_dict[model_b][model_a]['win'] += 1
    elif 'tie' in winner:
        win_dict[model_a][model_b]['tie'] += 1
        win_dict[model_b][model_a]['tie'] += 1


result_dict = {}

for target_model in args.model_name_list:
  
    
    
    with open(f'voting_output/{target_model}_{args.rigging_mode}_acc_{args.classifier_acc}_prob_dec_{args.beta}.json') as f:
        manipulated_battle_dict = json.load(f)
        
    start_list = np.random.choice(range(0,len(manipulated_battle_dict)-args.vote_num_per_user), args.users_num, replace=False)
   
         
    
    count = 0
    detect_users_num = 0
    for start_idx in start_list:
        for _ in range(args.random_round):
            p_x_adv = 0
            p_x_real = 0
            key_list = []
            for idx, key_idx in enumerate(manipulated_battle_dict):
                if idx < start_idx:
                    continue
                if len(key_list) == args.vote_num_per_user:
                    break
                
                model_a  = manipulated_battle_dict[key_idx]['model_a']
                model_b  = manipulated_battle_dict[key_idx]['model_b']
                adv_decision = manipulated_battle_dict[key_idx]['winner']
            
                
                if (win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie']) == 0 :
                    continue
                win_rate = win_dict[model_a][model_b]['win']/(win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie'])
                lose_rate = win_dict[model_a][model_b]['lose']/(win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie'])
                tie_rate = win_dict[model_a][model_b]['tie']/(win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie'])

                if win_rate==0 or lose_rate==0 or tie_rate==0 or adv_decision not in ['model_a', 'model_b', 'tie']:
                    continue
               

                if adv_decision == 'model_a':
                    p_x_adv += math.log(win_rate) * -2
                elif adv_decision == 'model_b':
                    p_x_adv += math.log(lose_rate) * -2
                elif 'tie' in adv_decision:
                    p_x_adv += math.log(tie_rate) * -2

                real_decision = np.random.choice(['model_a', 'model_b', 'tie'], p=[win_rate, lose_rate, 1-win_rate-lose_rate])

                if real_decision == 'model_a':
                    p_x_real += math.log(win_rate) * -2
                elif real_decision == 'model_b':
                    p_x_real += math.log(lose_rate) * -2
                elif 'tie' in real_decision:
                    p_x_real += math.log(tie_rate) * -2

                key_list.append(key_idx)

            # print(p_x_real, p_x_adv)
            if p_x_real > p_x_adv:
                count += 1
        if count/args.random_round < 0.01:
            detect_users_num += 1
    
    result_dict[target_model] = detect_users_num/len(start_list)

    os.makedirs('voting_output/detect_users', exist_ok=True)
    with open(f'voting_output/detect_users/{args.rigging_mode}_vote_num_{args.vote_num_per_user}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    