from collections import defaultdict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math, copy
import torch
import numpy as np
import pandas as pd
import plotly.express as px
pd.options.display.float_format = '{:.2f}'.format
from tqdm import tqdm

import argparse
import os
import cudf
import json

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--mode', type=str, default='diff_greedy')
parser.add_argument('--model_num', type=int, default=129)
parser.add_argument('--users_num', type=int, default=10)
parser.add_argument('--vote_num_per_user', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--random_round', type=int, default=100)



args = parser.parse_args()


def preety_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    
    df.index = df.index + 1
    return df


def preprocess_data():
   
    X_initial = np.load('data/initial_data_1.7m_0.9_portion.npy')
    Y_initial = np.load('data/initial_label_1.7m_0.9_portion.npy')
    win_matrix_initial = pd.read_csv('data/initial_win_matrix_1.7m_0.9_portion.csv', index_col=0)
    sample_weights = []
    for m_a in win_matrix_initial.index:
        for m_b in win_matrix_initial.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(win_matrix_initial.loc[m_a, m_b]) or math.isnan(win_matrix_initial.loc[m_b, m_a]):
                continue
            sample_weights.append(win_matrix_initial.loc[m_a, m_b])
            sample_weights.append(win_matrix_initial.loc[m_b, m_a])
    
    sample_weights = cudf.Series(sample_weights)
    X_initial = cudf.DataFrame(X_initial)
    Y_initial = cudf.Series(Y_initial)

    return X_initial, Y_initial, win_matrix_initial, sample_weights

def compute_mle_elo_dict(
    competition_list, X, Y, SCALE=400, BASE=10, INIT_RATING=1000, ptbl_win=None, sample_weights=None
):
   
    from cuml.linear_model import LogisticRegression
    for item in tqdm(competition_list):
       
        model_a = item['model_a']
        model_b = item['model_b']
        winner = item['winner']

        
        tmp_count = 0
        for m_a in ptbl_win.index:
            
            for m_b in ptbl_win.columns:
                if m_a == m_b:
                    continue

                if winner == 'model_a':
                    if model_a == m_a and model_b == m_b:
                        sample_weights[tmp_count] += 2
                    elif model_b == m_a and model_a == m_b:
                        sample_weights[tmp_count+1] += 2
                elif winner == 'model_b':
                    if model_a == m_a and model_b == m_b:
                       
                        sample_weights[tmp_count+1] += 2
                    elif model_b == m_a and model_a == m_b:
                       
                        sample_weights[tmp_count] += 2
                    
                elif 'tie' in winner:
                    if model_b == m_a and model_a == m_b:
                        sample_weights[tmp_count] += 1
                        sample_weights[tmp_count+1] += 1
                    if model_a == m_a and model_b == m_b:
                        sample_weights[tmp_count] += 1
                        sample_weights[tmp_count+1] += 1
                    
                tmp_count += 2

   
    
    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)
    
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_.to_numpy()[0] + INIT_RATING
   
    
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False), sample_weights




def get_rank(ranking, target_model):
    for idx, key in enumerate(ranking['Model']):
        if key == target_model:
            return idx + 1


# The above code is about the ranking moudle
# Manipulate the ranking with the following part
# --------------------------------------------------------------------------------------------------------------------


# load and filter targeted model names
with open('model_name_ranking_all.txt') as f:
    targeted_model_name = f.readlines()
    targeted_model_name = targeted_model_name[:args.model_num]
    for i in range(len(targeted_model_name)):
        targeted_model_name[i] = targeted_model_name[i].strip()
model_name_sorted = sorted(targeted_model_name)


K = 4
BASE = 10
SCALE = 400

random_choice_flag = args.mode # elo_greedy, random

print(f'Total {len(model_name_sorted)} models')



X_initial, Y_initial, win_matrix_initial, sample_weights_ori = preprocess_data()

print('Calculate Initial Rating')
elo_ratings, _ = compute_mle_elo_dict([], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=sample_weights_ori)
initial_rating = preety_print_model_ratings(elo_ratings)

print('---------------initial ranking---------------')
print(initial_rating)
print('---------------------------------------------')

win_dict = {}
for model in model_name_sorted:
    win_dict[model] = {}

for model1 in model_name_sorted:
    for model2 in model_name_sorted:
        if model1 != model2:
            win_dict[model1][model2] = {'win': 0, 'lose': 0, 'tie': 0}


with open(f'sanity_output_17/vh_1.7m.json') as f:
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

# exit(0)


model_name_list = ['llama-2-13b-chat', 'mistral-7b-instruct-v0.2', 'qwen1.5-14b-chat', 'vicuna-7b', 'gemma-7b-it', 'phi-3-mini-4k-instruct-june-2024']

result_dict = {}

for target_model in model_name_list:
  
    
    
    
    
    if args.mode == 'diff_greedy_defense':
        with open(f'detect_users/{target_model}_diff_greedy_defense_greedy_acc_1.0_random_acc_1.0_prob_dec_1.0_seed_{args.seed}.json') as f:
            manipulated_battle_dict = json.load(f)
    else:
        with open(f'sanity_output_17/{target_model}_{args.mode}_greedy_acc_1.0_random_acc_1.0_prob_dec_1.0_seed_{args.seed}.json') as f:
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
            
                # Adversarial Decision
                if (win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie']) == 0 :
                    continue
                win_rate = win_dict[model_a][model_b]['win']/(win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie'])
                lose_rate = win_dict[model_a][model_b]['lose']/(win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie'])
                tie_rate = win_dict[model_a][model_b]['tie']/(win_dict[model_a][model_b]['win'] + win_dict[model_a][model_b]['lose'] + win_dict[model_a][model_b]['tie'])

                if win_rate==0 or lose_rate==0 or tie_rate==0 or adv_decision not in ['model_a', 'model_b', 'tie']:
                    continue
                # if args.mode == 'diff_greedy_defense':
                #     if np.random.uniform() < 0.7 and model_a!= target_model and model_b != target_model:
                #         adv_decision = np.random.choice(['model_a', 'model_b', 'tie'], p=[win_rate, lose_rate, 1-win_rate-lose_rate])

                if args.mode == 'normal_voting':
                    if model_a!= target_model and model_b != target_model:
                        adv_decision = np.random.choice(['model_a', 'model_b', 'tie'], p=[win_rate, lose_rate, 1-win_rate-lose_rate])

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
    print(target_model, detect_users_num/len(start_list))
    result_dict[target_model] = detect_users_num/len(start_list)

    os.makedirs('detect_users', exist_ok=True)
    with open(f'detect_users/{args.mode}_votelength_{args.vote_num_per_user}_seed_{args.seed}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    