import os
import copy
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
from utils import preety_print_model_ratings, get_rank, get_battle_pair, preprocess_data, compute_mle_elo_dict

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--rigging_mode', type=str, default='omni_bt_diff')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--vote_num', type=int, default=20000)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--classifier_acc', type=float, default=1.0)
parser.add_argument('--model_name_list', nargs='+', default=['gpt-4o-2024-05-13'])
args = parser.parse_args()

K = 4
BASE = 10
SCALE = 400


# --------------------------------------------Initialize the rigging environment--------------------------------------------

# Load the initial data
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
   
if args.rigging_mode == 'omni_bt_diff':
    diff_weight = 1
else:
    diff_weight = 0


win_dict = {}
for model in model_name_sorted:
    win_dict[model] = {}

# Calculate the historical win rate between each pair of models
for model_a in model_name_sorted:
    for model_b in model_name_sorted:
        if model_a != model_b:
            win_dict[model_a][model_b] = {'win': 0, 'lose': 0, 'tie': 0}


with open(f'data/vh.json') as f:
    vh_dict = json.load(f)
print(len(vh_dict.keys()))
for key_idx in vh_dict:
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

# --------------------------------------------Rigging simulation--------------------------------------------


for target_model in args.model_name_list:
    np.random.seed(args.seed)
    battle_dict = {}
    model_name_sorted_prob = {}
    rank_list = [get_rank(initial_ranking, target_model)]

    # Generate the sampling distribution for each model, \beta controls the marginal probability of sampling the target model
    for model in model_name_sorted:
        if model == target_model:
            model_weight_tmp = 1 * args.beta
        else:
            model_weight_tmp = 1 + (1 - args.beta)/(len(model_name_sorted) - 1)
        model_name_sorted_prob[model] = model_weight_tmp
   

    print(target_model)
    sample_weights_tmp, sample_weights_tmp_noise = copy.deepcopy(sample_weights_ori), copy.deepcopy(sample_weights_ori)
    final_ranking, final_ranking_noise = copy.deepcopy(initial_ranking), copy.deepcopy(initial_ranking)
    
    for idx in range(args.vote_num):
        
        model_a, model_b = get_battle_pair(model_name_sorted, model_name_sorted_prob)
        current_battle = {'model_a':model_a, 'model_b':model_b, 'winner':None}
        
        # For a fair comparison, we ensure the sampled battles are the same for each rigging strategy
        vote_var = np.random.uniform()
        acc_var = np.random.uniform()
        decision_var = np.random.uniform()
        pred_model_a_var = np.random.choice(len(model_name_sorted)-1)
        pred_model_b_var = np.random.choice(len(model_name_sorted)-1)

        # Rigging strategies
        if args.rigging_mode == 't_random' or args.rigging_mode == 't_tie' or args.rigging_mode == 't_normal':
            if (model_b == target_model or model_a == target_model) and acc_var <= args.classifier_acc:
                decision = 'model_b' if model_b == target_model else 'model_a'
            else:
                if args.rigging_mode == 't_random':
                    if vote_var < 1/4:
                        decision = 'model_a'
                    elif vote_var < 2/4:
                        decision = 'model_b'
                    elif vote_var < 3/4:
                        decision = 'tie'
                    elif vote_var < 1:
                        decision = 'remove'
                elif args.rigging_mode == 't_tie':
                    decision = 'tie'
                elif args.rigging_mode == 't_normal':
                    if acc_var > args.classifier_acc:
                        tmp_model_name_sorted = copy.deepcopy(model_name_sorted)
                        tmp_model_name_sorted.pop(tmp_model_name_sorted.index(model_a))
                        tmp_model_name_sorted.pop(tmp_model_name_sorted.index(model_b))
                        # pred_model_a = np.random.choice(tmp_model_name_sorted)
                        # pred_model_b = np.random.choice(tmp_model_name_sorted)
                        pred_model_a = tmp_model_name_sorted[pred_model_a_var]
                        pred_model_b = tmp_model_name_sorted[pred_model_b_var]
                    else:
                        pred_model_a, pred_model_b = model_a, model_b

                    win_rate_base = win_dict[pred_model_a][pred_model_b]['win'] + win_dict[pred_model_a][pred_model_b]['lose'] + win_dict[pred_model_a][pred_model_b]['tie']
                    if win_rate_base != 0 :
                        win_rate = win_dict[pred_model_a][pred_model_b]['win']/win_rate_base
                        lose_rate = win_dict[pred_model_a][pred_model_b]['lose']/win_rate_base
                        tie_rate = win_dict[pred_model_a][pred_model_b]['tie']/win_rate_base

                        # decision = np.random.choice(['model_a', 'model_b', 'tie'], p=[win_rate, lose_rate, tie_rate])
                        if decision_var < win_rate:
                            decision = 'model_a'
                        elif decision_var < win_rate + lose_rate:
                            decision = 'model_b'
                        else:
                            decision = 'tie'
                    else:
                        win_prob_a = 1 / (1 + BASE ** ((initial_rating[pred_model_b] - initial_rating[pred_model_a]) / SCALE))

                        if vote_var > win_prob_a:
                            decision = 'model_b'
                        else:
                            decision = 'model_a'
            
            current_battle['winner'] = decision
            final_ranking, sample_weights_tmp = compute_mle_elo_dict([current_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
            final_ranking = preety_print_model_ratings(final_ranking)
        

        elif args.rigging_mode == 't_abstain':
            if (model_b == target_model or model_a == target_model) and acc_var <= args.classifier_acc:
                
                decision = 'model_b' if model_b == target_model else 'model_a'
                current_battle['winner'] = decision
                final_ranking, sample_weights_tmp = compute_mle_elo_dict([current_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
                final_ranking = preety_print_model_ratings(final_ranking)
            else:
                decision = 'remove'
        

        
        elif args.rigging_mode == 'omni_on':

            pred_model_a = model_a
            pred_model_b = model_b
            
            if acc_var > args.classifier_acc:
                tmp_model_name_sorted = copy.deepcopy(model_name_sorted)
                tmp_model_name_sorted.pop(tmp_model_name_sorted.index(model_a))
                tmp_model_name_sorted.pop(tmp_model_name_sorted.index(model_b))
                # pred_model_a = np.random.choice(tmp_model_name_sorted)
                # pred_model_b = np.random.choice(tmp_model_name_sorted)
                pred_model_a = tmp_model_name_sorted[pred_model_a_var]
                pred_model_b = tmp_model_name_sorted[pred_model_b_var]
                
            if pred_model_a == pred_model_b:
                rank_list.append(rank_list[-1])
                battle_dict[f'idx_{idx}'] = {'model_a': model_a, 'model_b': model_b, 'winner': 'remove'}
                continue

         
            
            reward_list = []

            ra = initial_rating[pred_model_a]
            rb = initial_rating[pred_model_b]
            rt = initial_rating[target_model] 
            ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
            eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))

            
            # case 1
            ra1 = ra + K * eb
            rb1 = rb - K * eb
            reward = 1 / (1 + BASE ** ((ra1 - rt) / SCALE)) + 1 / (1 + BASE ** ((rb1 - rt) / SCALE))

            reward_list.append(reward) 

            # case 2            
            ra1 = ra - K * ea
            rb1 = rb + K * ea
            reward = 1 / (1 + BASE ** ((ra1 - rt) / SCALE)) + 1 / (1 + BASE ** ((rb1 - rt) / SCALE))

            reward_list.append(reward) 

            # case 3
            ra1 = ra - 0.5 * K * (ea - eb)
            rb1 = rb + 0.5 * K * (ea - eb)
            reward = 1 / (1 + BASE ** ((ra1 - rt) / SCALE)) + 1 / (1 + BASE ** ((rb1 - rt) / SCALE))

            reward_list.append(reward) 

            # case 4
            ra1 = ra
            rb1 = rb
            reward = 1 / (1 + BASE ** ((ra1 - rt) / SCALE)) + 1 / (1 + BASE ** ((rb1 - rt) / SCALE))

            reward_list.append(reward) 
        

            if reward_list.index(max(reward_list)) == 0:
                decision = 'model_a'
            elif reward_list.index(max(reward_list)) == 1:
                decision = 'model_b'
            elif reward_list.index(max(reward_list)) == 2:
                decision = 'tie'
            elif reward_list.index(max(reward_list)) == 3:
                decision = 'remove'
            
            if pred_model_a == target_model:
                decision = 'model_a'
            
            if pred_model_b == target_model:
                decision = 'model_b'

            current_battle['winner'] = decision
            final_ratings, sample_weights_tmp = compute_mle_elo_dict([current_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
            final_ranking = preety_print_model_ratings(final_ratings)


        elif args.rigging_mode == 'omni_bt_diff' or args.rigging_mode == 'omni_bt_abs':
            
            reward_list = []
            tmp_ranking_list = []
            tmp_weights_list = []

            pred_model_a = model_a
            pred_model_b = model_b
            
            if acc_var > args.classifier_acc:
                tmp_model_name_sorted = copy.deepcopy(model_name_sorted)
                tmp_model_name_sorted.pop(tmp_model_name_sorted.index(model_a))
                tmp_model_name_sorted.pop(tmp_model_name_sorted.index(model_b))
                # pred_model_a = np.random.choice(tmp_model_name_sorted)
                # pred_model_b = np.random.choice(tmp_model_name_sorted)
                pred_model_a = tmp_model_name_sorted[pred_model_a_var]
                pred_model_b = tmp_model_name_sorted[pred_model_b_var]


            if pred_model_a == pred_model_b:
                rank_list.append(rank_list[-1])
                battle_dict[f'idx_{idx}'] = {'model_a': model_a, 'model_b': model_b, 'winner': 'remove'}
                continue

            for key in final_ranking_noise['Model'].keys():
                if final_ranking_noise.loc[key, 'Model'] == target_model:
                    
                    if key != 1:
                        tmp_anchor_model = final_ranking_noise.loc[key-1, 'Model']
                        tmp_anchor_model_rating = final_ranking_noise.loc[key-1, 'Elo rating']
                    else:
                        tmp_anchor_model = final_ranking_noise.loc[key+1, 'Model']
                        tmp_anchor_model_rating = final_ranking_noise.loc[key+1, 'Elo rating']

                    remove_rating = final_ranking_noise.loc[key, 'Elo rating'] - tmp_anchor_model_rating * diff_weight

                
            else:
                
                for key in final_ranking['Model'].keys():
                    if final_ranking.loc[key, 'Model'] == target_model:
                        
                        if key != 1:
                            tmp_anchor_model = final_ranking.loc[key-1, 'Model']
                            tmp_anchor_model_rating = final_ranking.loc[key-1, 'Elo rating']
                        else:
                            tmp_anchor_model = final_ranking.loc[key+1, 'Model']
                            tmp_anchor_model_rating = final_ranking.loc[key+1, 'Elo rating']

                        remove_rating = final_ranking.loc[key, 'Elo rating'] - tmp_anchor_model_rating * diff_weight
            

                
           
            for tmp_vote in ['model_a', 'model_b', 'tie']:
                tmp_battle = {'model_a':pred_model_a, 'model_b':pred_model_b, 'winner': tmp_vote}
                
                if args.classifier_acc < 1:
                    tmp_ratings, tmp_weights = compute_mle_elo_dict([tmp_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp_noise))
                    tmp_ranking = preety_print_model_ratings(tmp_ratings)
                else:
                    tmp_ratings, tmp_weights = compute_mle_elo_dict([tmp_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
                    tmp_ranking = preety_print_model_ratings(tmp_ratings)
    

                tmp_ranking_list.append(copy.deepcopy(tmp_ranking))
                tmp_weights_list.append(copy.deepcopy(tmp_weights))

                for _, key in enumerate(tmp_ranking['Model'].keys()):
                    if tmp_ranking.loc[key, 'Model'] == tmp_anchor_model:
                        tmp_anchor_model_rating = tmp_ranking.loc[key, 'Elo rating']
                    if tmp_ranking.loc[key, 'Model'] == target_model:
                        tmp_target_model_rating = tmp_ranking.loc[key, 'Elo rating']


                reward_list.append(tmp_target_model_rating - tmp_anchor_model_rating * diff_weight)
                
            reward_list.append(remove_rating)
            
            if reward_list.index(max(reward_list)) == 0:
                decision = 'model_a'

            elif reward_list.index(max(reward_list)) == 1:
                decision = 'model_b'
                
            elif reward_list.index(max(reward_list)) == 2:
                decision = 'tie'
                
            elif reward_list.index(max(reward_list)) == 3:
                decision = 'remove'

            
            if args.classifier_acc < 1:
                current_battle['winner'] = decision
                final_ratings, sample_weights_tmp = compute_mle_elo_dict([current_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
                final_ranking = preety_print_model_ratings(final_ratings)
                if decision == 'model_a':
                    final_ranking_noise = tmp_ranking_list[0]
                    sample_weights_tmp_noise = tmp_weights_list[0]
                    
                elif decision == 'model_b':
                    final_ranking_noise = tmp_ranking_list[1]
                    sample_weights_tmp_noise = tmp_weights_list[1]
                    
                elif decision == 'tie':
                    final_ranking_noise = tmp_ranking_list[2]
                    sample_weights_tmp_noise = tmp_weights_list[2]
                    
            else:
                if decision == 'model_a':
                    final_ranking = tmp_ranking_list[0]
                    sample_weights_tmp = tmp_weights_list[0]
                    
                elif decision == 'model_b':
                    final_ranking = tmp_ranking_list[1]
                    sample_weights_tmp = tmp_weights_list[1]
                    
                elif decision == 'tie':
                    final_ranking = tmp_ranking_list[2]
                    sample_weights_tmp = tmp_weights_list[2]
                    
            
            # assert reward_list[0] != reward_list[1]
            del tmp_ranking_list, tmp_ranking, tmp_weights_list

        
        battle_dict[f'idx_{idx}'] = {'model_a': model_a, 'model_b': model_b, 'winner': decision}
        
        rank_list.append(get_rank(final_ranking, target_model))

    
        if model_a == target_model or model_b == target_model:
            print(f'Battle idx: {idx} | Mode: {args.rigging_mode} | decision: {decision} | Rank: {get_rank(final_ranking, target_model)} | {model_a} vs {model_b} (target)')
        else:
            print(f'Battle idx: {idx} | Mode: {args.rigging_mode} | decision: {decision} | Rank: {get_rank(final_ranking, target_model)} | {model_a} vs {model_b}')
        

   
    os.makedirs('ranking_output/', exist_ok=True) 
    os.makedirs('voting_output/', exist_ok=True)

    np.save(f'ranking_output/{target_model}_{args.rigging_mode}_acc_{args.classifier_acc}_prob_dec_{args.beta}.npy', np.array(rank_list))
    with open(f'voting_output/{target_model}_{args.rigging_mode}_acc_{args.classifier_acc}_prob_dec_{args.beta}.json', 'w') as f:
        json.dump(battle_dict, f, indent=4)




        

   
        
