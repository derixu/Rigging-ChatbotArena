from collections import defaultdict
import math
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
from tqdm import tqdm
import cudf

# Global variables
K = 4
BASE = 10
SCALE = 400

def preety_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    
    df.index = df.index + 1
    return df


def preprocess_data(X_initial_path, Y_initial_path, win_matrix_initial_path):
  
    X_initial = np.load(X_initial_path)
    Y_initial = np.load(Y_initial_path)
    win_matrix_initial = pd.read_csv(win_matrix_initial_path, index_col=0)
    sample_weights = []
    for m_a in win_matrix_initial.index:
        for m_b in win_matrix_initial.columns:
            if m_a == m_b:
                continue
            if math.isnan(win_matrix_initial.loc[m_a, m_b]) or math.isnan(win_matrix_initial.loc[m_b, m_a]):
                continue
            sample_weights.append(win_matrix_initial.loc[m_a, m_b])
            sample_weights.append(win_matrix_initial.loc[m_b, m_a])
    
    sample_weights = cudf.Series(sample_weights)
    X_initial = cudf.DataFrame(X_initial)
    Y_initial = cudf.Series(Y_initial)

    return X_initial, Y_initial, win_matrix_initial, sample_weights

def compute_mle_elo_dict(
    competition_list, X, Y, SCALE=400, INIT_RATING=1000, ptbl_win=None, sample_weights=None
):
   
    from cuml.linear_model import LogisticRegression
    for item in competition_list:
       
        model_a = item['model_a']
        model_b = item['model_b']
        winner = item['winner']

        
        count = 0
        for m_a in ptbl_win.index:
            for m_b in ptbl_win.columns:
                if m_a == m_b:
                    continue

                if winner == 'model_a':
                    if model_a == m_a and model_b == m_b:
                        sample_weights[count] += 2
                    elif model_b == m_a and model_a == m_b:
                        sample_weights[count+1] += 2

                elif winner == 'model_b':
                    if model_a == m_a and model_b == m_b:
                        sample_weights[count+1] += 2
                    elif model_b == m_a and model_a == m_b:
                        sample_weights[count] += 2
                    
                elif 'tie' in winner:
                    if model_b == m_a and model_a == m_b:
                        sample_weights[count] += 1
                        sample_weights[count+1] += 1
                    if model_a == m_a and model_b == m_b:
                        sample_weights[count] += 1
                        sample_weights[count+1] += 1
                    
                count += 2

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_.to_numpy()[0] + INIT_RATING
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False), sample_weights

def get_rank(ranking, target_model):
    for idx, key in enumerate(ranking['Model']):
        if key == target_model:
            return idx + 1

def get_sample_weight(model, sampling_weights):
   
    weight = sampling_weights.get(model, 0)
    return weight

def get_battle_pair(models, sampling_weights):
    if len(models) == 1:
        return models[0], models[0]

    model_weights = []
    for model in models:
        weight = get_sample_weight(
            model, sampling_weights
        )
        model_weights.append(weight)
    total_weight = np.sum(model_weights)
    model_weights = model_weights / total_weight
    
    chosen_idx = np.random.choice(len(models), p=model_weights)
    chosen_model = models[chosen_idx]

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
       
        weight = get_sample_weight(model, sampling_weights)
       
        rival_models.append(model)
        rival_weights.append(weight)
    
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]
    return chosen_model, rival_model
    
def get_battle_prob(model_a, model_b, models, sampling_weights):
    model_weights = []
    for model in models:
        weight = get_sample_weight(
            model, sampling_weights
        )
        if model == model_a:
            target_weight_a = weight
        if model == model_b:
            target_weight_b = weight
        model_weights.append(weight)

    total_weight = np.sum(model_weights)
    prob_a = target_weight_a / total_weight * target_weight_b / (total_weight-target_weight_a)
    prob_b = target_weight_b / total_weight * target_weight_a / (total_weight-target_weight_b)
    return prob_a + prob_b