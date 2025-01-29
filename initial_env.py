import os
import numpy as np
import pandas as pd
import requests
pd.options.display.float_format = '{:.2f}'.format

from utils import preety_print_model_ratings, initialize_vh_vo
from tqdm import tqdm
import json



# Load the JSON data from the local file
if not os.path.exist('data/local_file_name.json'):
    url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json"
    response = requests.get(url)

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


with open(f'data/vo.json', 'w') as f:
    json.dump(battle_vo_dict, f, indent=4)

with open(f'data/vh.json', 'w') as f:
    json.dump(battle_vh_dict, f, indent=4)


elo_ratings, _ = initialize_vh_vo(model_name_sorted, battle_vh_list)
initial_ranking = preety_print_model_ratings(elo_ratings)


print(initial_ranking)

        

   
        