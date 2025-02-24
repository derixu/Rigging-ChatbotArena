import os
import numpy as np
import pandas as pd
import requests
pd.options.display.float_format = '{:.2f}'.format

from utils import preety_print_model_ratings, initialize_vh_vo
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--classifier', action='store_true')
args = parser.parse_args()


# Load the JSON data from the local file
if not os.path.exists('data/local_file_name.json'):
    url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json"
    response = requests.get(url)
    os.makedirs('data', exist_ok=True)

    with open('data/local_file_name.json', 'wb') as file:
        file.write(response.content)

with open('data/local_file_name.json', 'r') as file:
    battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])


# Filter targeted battles

battles = battles[battles["anony"] == True]
# battles = battles[battles["language"] == 'English']
print("Before dedup: ", len(battles))
battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
print("After dedup: ", len(battles))
battles = battles.sort_values(ascending=True, by=["tstamp"])

if args.classifier:
    model_name_classifier = ['llama-2-7b-chat', 'llama-2-13b-chat', 'llama-3-8b-instruct', 'command-r', 'gpt-4o-mini-2024-07-18',\
                 'mistral-7b-instruct-v0.2', 'mistral-7b-instruct', 'gemma-2-27b-it', 'gemma-2b-it', 'gemma-2-9b-it', \
                     'qwen1.5-7b-chat', 'qwen1.5-14b-chat', 'starling-lm-7b-alpha', 'starling-lm-7b-beta', 'yi-34b-chat','yi-1.5-34b-chat',\
                        'chatglm3-6b', 'zephyr-7b-beta', 'zephyr-7b-alpha', 'phi-3-small-8k-instruct', \
                            'vicuna-7b', 'mpt-7b-chat', 'openchat-3.5', 'wizardlm-13b', 'solar-10.7b-instruct-v1.0']
    
    battles = battles[battles["language"] == 'English']
    battle_list = []
    for model_id in model_name_classifier:
        battle_list.append(battles[battles["model_a"] == model_id])
    battles = pd.concat(battle_list)


    battle_list = []
    for model_id in model_name_classifier:
        battle_list.append(battles[battles["model_b"] == model_id])
    battles = pd.concat(battle_list)

model_name_sorted = []
for model in battles["model_a"]:
    if model not in model_name_sorted:
        model_name_sorted.append(model)
model_name_sorted = sorted(model_name_sorted)




battle_all_list = []

for idx, key in tqdm(enumerate(battles['model_a'].keys())):
    model_a = battles.loc[key, 'model_a'] 
    model_b = battles.loc[key, 'model_b'] 
    winner = battles.loc[key, 'winner'] 
    tokens_a = battles.loc[key, 'conv_metadata']['sum_assistant_a_tokens']
    tokens_b = battles.loc[key, 'conv_metadata']['sum_assistant_b_tokens']
   
    battle_all_list.append({'model_a':model_a, 'model_b':model_b, 'winner':winner, 'tokens_a':tokens_a, 'tokens_b':tokens_b})
   



battle_vo_list = []
battle_vh_list = []

index_list = np.random.choice([x for x in range(len(battle_all_list))], int(0.9*len(battle_all_list)), replace=False)
for idx in range(len(battle_all_list)):
    if idx not in index_list:
        battle_vo_list.append(battle_all_list[idx])
    else:
        battle_vh_list.append(battle_all_list[idx])
    
print(len(battle_vo_list))
print(len(battle_vh_list))

battle_vo_dict = {}
for idx,item in enumerate(battle_vo_list):
    battle_vo_dict[idx] = {'model_a':item['model_a'], 'model_b':item['model_b'], 'winner':item['winner'],'tokens_a':tokens_a, 'tokens_b':tokens_b}

battle_vh_dict = {}
for idx,item in enumerate(battle_vh_list):
    battle_vh_dict[idx] = {'model_a':item['model_a'], 'model_b':item['model_b'], 'winner':item['winner'],'tokens_a':tokens_a, 'tokens_b':tokens_b}

if args.classifier:
    with open(f'data/vo_classifier.json', 'w') as f:
        json.dump(battle_vo_dict, f, indent=4)

    with open(f'data/vh_classifier.json', 'w') as f:
        json.dump(battle_vh_dict, f, indent=4)
else:
    with open(f'data/vo.json', 'w') as f:
        json.dump(battle_vo_dict, f, indent=4)

    with open(f'data/vh.json', 'w') as f:
        json.dump(battle_vh_dict, f, indent=4)


elo_ratings, _ = initialize_vh_vo(model_name_sorted, battle_vh_list,classifier=args.classifier)
initial_ranking = preety_print_model_ratings(elo_ratings)


print(initial_ranking)

        

   
        
