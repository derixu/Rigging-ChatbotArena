import os
import copy
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
import torch
import json
import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import preety_print_model_ratings, get_rank, preprocess_data, compute_mle_elo_dict, get_battle_prob, get_battle_pair

parser = argparse.ArgumentParser()
parser.add_argument('--rigging_mode', type=str, default='omni_bt_diff')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--battle_num', type=int, default=1000)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--model_path', type=str)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--dataset_name', type=str, default='hc3')
parser.add_argument('--model_name_list', nargs='+', default=['phi-3-mini-4k-instruct-june-2024'])

args = parser.parse_args()

model_name_classifier_map = ['llama-2-7b-chat', 'llama-2-13b-chat', 'llama-3-8b-instruct', 'command-r', 'gpt-4o-mini-2024-07-18',\
                 'mistral-7b-instruct-v0.2', 'mistral-7b-instruct', \
                  'gemma-2-27b-it', 'gemma-2b-it', 'gemma-2-9b-it',\
                     'qwen1.5-7b-chat', 'qwen1.5-14b-chat', 'starling-lm-7b-alpha', 'starling-lm-7b-beta', 'yi-34b-chat','yi-1.5-34b-chat',\
                        'chatglm3-6b', 'zephyr-7b-beta', 'zephyr-7b-alpha', 'phi-3-small-8k-instruct', \
                            'vicuna-7b', 'mpt-7b-chat', 'openchat-3.5', 'wizardlm-13b', 'solar-10.7b-instruct-v1.0']

from_file_to_arena_map = {'llama-2-7b-chat': 'Llama-2-7b-chat-hf', 'llama-2-13b-chat': 'Llama-2-13b-chat-hf', 'llama-3-8b-instruct': 'Meta-Llama-3-8B-Instruct', \
             'command-r': 'c4ai-command-r-v01', 'gpt-4o-mini-2024-07-18': 'gpt-4o-mini-2024-07-18',\
                 'mistral-7b-instruct-v0.2': 'Mistral-7B-Instruct-v0.2', 'mistral-7b-instruct': 'Mistral-7B-Instruct-v0.1', \
                  'gemma-2-27b-it': 'gemma-2-27b-it', 'gemma-2b-it': 'gemma-2b-it', 'gemma-2-9b-it': 'gemma-2-9b-it',\
                     'qwen1.5-7b-chat': 'Qwen1.5-7B-Chat', 'qwen1.5-14b-chat': 'Qwen1.5-14B-Chat', 'starling-lm-7b-alpha': 'Starling-LM-7B-alpha', \
                        'starling-lm-7b-beta': 'Starling-LM-7B-beta', 'yi-34b-chat': 'Yi-34B-Chat','yi-1.5-34b-chat': 'Yi-1.5-34B-Chat',\
                        'chatglm3-6b': 'chatglm3-6b', 'zephyr-7b-beta': 'zephyr-7b-beta', 'zephyr-7b-alpha': 'zephyr-7b-alpha', 'phi-3-small-8k-instruct': 'Phi-3-small-8k-instruct', \
                            'vicuna-7b': 'vicuna-7b-v1.3', 'mpt-7b-chat': 'mpt-7b-chat', 'openchat-3.5': 'openchat_3.5', 'wizardlm-13b': 'WizardLM-13B-V1.2', \
                                'solar-10.7b-instruct-v1.0': 'SOLAR-10.7B-Instruct-v1.0', 'phi-3-medium-4k-instruct': 'phi-3-medium-4k-instruct',\
                                'dolphin-2.2.1-mistral-7b': 'dolphin-2.2.1-mistral-7b', 'openhermes-2.5-mistral-7b': 'OpenHermes-2.5-Mistral-7B', \
                                    'vicuna-13b': 'vicuna-13b-v1.3', 'gpt-3.5-turbo-0613': 'gpt-3.5-turbo'}



K = 4
BASE = 10
SCALE = 400


# Initialize the rigging environment
X_initial, Y_initial, win_matrix_initial, sample_weights_ori = preprocess_data('data/data_x_classifier.npy', 'data/data_y_classifier.npy','data/vh_win_matrix_classifier.csv')
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

   

response_list = {}
kwargs = dict(max_length=512, truncation=True)
tokenizer = AutoTokenizer.from_pretrained(args.rigging_model_path, **kwargs)
text_classifier = AutoModelForSequenceClassification.from_pretrained(args.rigging_model_path, num_labels=len(model_name_classifier_map), device_map='cuda')
text_classifier.eval()

if args.eval:
    start_idx = 4000
else:
    start_idx = 0

for model_id in model_name_sorted:
    tmp_response_list = {}
    model_id_raw = from_file_to_arena_map[model_id] 
  
    file_name = f'./classifier/training_data_classifier/{args.dataset_name}/{model_id_raw}.json'
    with open(file_name) as f:
        data = json.load(f)
        print(model_id)

        for i in range(start_idx,start_idx+args.battle_num):
            
            response = data[f'id_{i}']['response'].strip()
            response = response.replace('***', '')
            response = response.replace('**', '')
            response = response.replace('#', '')
            response = response.replace('-', '')
            tokenized_response = tokenizer.encode(response)
            tmp_response_list[i] = {'response': torch.tensor(tokenized_response[:512]).reshape(1, -1), 'length': len(tokenized_response)}
    
    response_list[model_id] = tmp_response_list



if args.rigging_mode == 'diff_greedy':
    alpha_weight = 1
else:
    alpha_weight = 0



for target_model in args.rigging_model_name_list:

    battle_dict = {}
    model_name_sorted_prob = {}
    tmp_sum = 0
    for model in model_name_sorted:
        if model == target_model:
            model_weight_tmp = 1 * args.beta
        else:
            model_weight_tmp = 1 + (1 - args.beta)/(len(model_name_sorted) - 1)
        tmp_sum += model_weight_tmp
        model_name_sorted_prob[model] = model_weight_tmp
    for tmp_key in model_name_sorted_prob:
        model_name_sorted_prob[tmp_key] /= tmp_sum

    count = 0
    for tmp_id, model_a_tmp in enumerate(model_name_sorted):
        for model_b_tmp in model_name_sorted:
            if model_b_tmp != model_a_tmp:
                tmp_prob = get_battle_prob(model_a_tmp, model_b_tmp, model_name_sorted, model_name_sorted_prob)
                count += tmp_prob

    

    np.random.seed(args.seed)
    
    print(target_model)
    sample_weights_tmp = copy.deepcopy(sample_weights_ori)
    sample_weights_tmp_noise = copy.deepcopy(sample_weights_ori)
    

    count = 0
    final_ranking = copy.deepcopy(initial_ranking)
    final_ranking_noise = copy.deepcopy(initial_ranking)

    rank_list = [get_rank(initial_ranking, target_model)]



    for idx in range(args.battle_num):
        real_model_a, real_model_b = get_battle_pair(model_name_sorted, model_name_sorted_prob)
        random_index = np.random.uniform()
        current_battle = {'model_a':real_model_a, 'model_b':real_model_b, 'winner':None}
       

        if args.rigging_mode == 't_abstain':
            
            if real_model_a == target_model or real_model_b == target_model:
            
                decision = 'model_b' if real_model_b == target_model else 'model_a'
                current_battle['winner'] = decision
                final_ratings, sample_weights_tmp = compute_mle_elo_dict([current_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
                final_ranking = preety_print_model_ratings(final_ratings)       
            else:
                decision = 'remove'
           

        elif args.rigging_mode == 'omni_on':
            model_a_response = response_list[real_model_a][start_idx+idx]['response'].to('cuda')
            model_b_response = response_list[real_model_b][start_idx+idx]['response'].to('cuda')
            
            tokens_a = int(response_list[real_model_a][start_idx+idx]['length'])
            tokens_b = int(response_list[real_model_b][start_idx+idx]['length'])

            

            pred_model_a = model_name_classifier_map[int(text_classifier(model_a_response).logits.argmax(-1))]
            pred_model_b = model_name_classifier_map[int(text_classifier(model_b_response).logits.argmax(-1))]

            if pred_model_b == pred_model_a:
                
                rank_list.append(rank_list[-1])
                battle_dict[f'idx_{idx}'] = {'model_a': real_model_a, 'model_b': real_model_b, 'decision': decision, 'tokens_a':tokens_a, 'tokens_b':tokens_b}
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
        
            model_a_response = response_list[real_model_a][start_idx+idx]['response'].to('cuda')
            model_b_response = response_list[real_model_b][start_idx+idx]['response'].to('cuda')
            tokens_a = int(response_list[real_model_a][start_idx+idx]['length'])
            tokens_b = int(response_list[real_model_b][start_idx+idx]['length'])
           
            pred_model_a = model_name_classifier_map[int(text_classifier(model_a_response).logits.argmax(-1))]
            pred_model_b = model_name_classifier_map[int(text_classifier(model_b_response).logits.argmax(-1))]
            if pred_model_b == pred_model_a:
                rank_list.append(rank_list[-1])
                battle_dict[f'idx_{idx}'] = {'model_a': real_model_a, 'model_b': real_model_b, 'decision': decision, 'tokens_a':tokens_a, 'tokens_b':tokens_b}
                continue
        
            
            
            reward_list = []
            tmp_rating_list = []
            tmp_weights_list = []


            for _, key in enumerate(final_ranking_noise['Model'].keys()):
                if final_ranking_noise.loc[key, 'Model'] == target_model:
                    
                    if key != 1:
                        tmp_anchor_model = final_ranking_noise.loc[key-1, 'Model']
                        tmp_anchor_model_rating = final_ranking_noise.loc[key-1, 'Elo rating']
                    else:
                        tmp_anchor_model = final_ranking_noise.loc[key+1, 'Model']
                        tmp_anchor_model_rating = final_ranking_noise.loc[key+1, 'Elo rating']

                    remove_rating = final_ranking_noise.loc[key, 'Elo rating'] - tmp_anchor_model_rating * alpha_weight
            
            
            for tmp_vote in ['model_a', 'model_b', 'tie']:
                tmp_battle = {'model_a':pred_model_a, 'model_b':pred_model_b, 'winner': tmp_vote}
                
                
                tmp_ratings, tmp_weights = compute_mle_elo_dict([tmp_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp_noise))
                tmp_ranking = preety_print_model_ratings(tmp_ratings)

                tmp_weights_list.append(copy.deepcopy(tmp_weights))
                tmp_rating_list.append(copy.deepcopy(tmp_ranking))
                
                for _, key in enumerate(tmp_ranking['Model'].keys()):
                    if tmp_ranking.loc[key, 'Model'] == tmp_anchor_model:
                        tmp_anchor_model_rating = tmp_ranking.loc[key, 'Elo rating']
                    if tmp_ranking.loc[key, 'Model'] == target_model:
                        tmp_target_model_rating = tmp_ranking.loc[key, 'Elo rating']


                reward_list.append(tmp_target_model_rating - tmp_anchor_model_rating * alpha_weight)
                
            reward_list.append(remove_rating)
            
            
            if reward_list.index(max(reward_list)) == 0:
                
                decision = 'model_a'
                final_ranking_noise = tmp_rating_list[0]
                sample_weights_tmp_noise = tmp_weights_list[0]
                
            elif reward_list.index(max(reward_list)) == 1:
                
                decision = 'model_b'
                final_ranking_noise = tmp_rating_list[1]
                sample_weights_tmp_noise = tmp_weights_list[1]
            elif reward_list.index(max(reward_list)) == 2:
            
                decision = 'tie'
                final_ranking_noise = tmp_rating_list[2]
                sample_weights_tmp_noise = tmp_weights_list[2]
            elif reward_list.index(max(reward_list)) == 3:
                decision = 'remove'

            current_battle['winner'] = decision
            
           
            del tmp_rating_list, tmp_ranking

            final_ratings, sample_weights_tmp = compute_mle_elo_dict([current_battle], X=X_initial, Y=Y_initial, ptbl_win=win_matrix_initial, sample_weights=copy.deepcopy(sample_weights_tmp))
            final_ranking = preety_print_model_ratings(final_ratings)

        if 'omni' in args.rigging_mode:
            battle_dict[f'idx_{idx}'] = {'model_a': real_model_a, 'model_b': real_model_b, 'decision': decision, 'tokens_a':tokens_a, 'tokens_b':tokens_b}
        else:
            battle_dict[f'idx_{idx}'] = {'model_a': real_model_a, 'model_b': real_model_b, 'decision': decision}
        

        rank_list.append(get_rank(final_ranking, target_model))
       

                
    
    
    os.makedirs('ranking_output_classifier/', exist_ok=True) 
    os.makedirs('voting_output_classifier', exist_ok=True)

    if args.eval:
        np.save(f'ranking_output_classifier/{target_model}_{args.rigging_mode}_prob_dec_{args.beta}_{args.dataset_name}_eval.npy', np.array(rank_list))
        with open(f'voting_output_classifier/{target_model}_{args.rigging_mode}_prob_dec_{args.beta}_{args.dataset_name}_eval.json', 'w') as f:
            json.dump(battle_dict, f, indent=4)
    else:
        np.save(f'ranking_output_classifier/{target_model}_{args.rigging_mode}_prob_dec_{args.beta}_{args.dataset_name}.npy', np.array(rank_list))
        with open(f'voting_output_classifier/{target_model}_{args.rigging_mode}_prob_dec_{args.beta}_{args.dataset_name}.json', 'w') as f:
            json.dump(battle_dict, f, indent=4)



        

   
        