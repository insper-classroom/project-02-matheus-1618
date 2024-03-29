from collections import defaultdict
import os
import pickle
import random
import requests
import time
import tqdm

from IPython.core.debugger import set_trace
import numpy as np
import pandas as pd
from pytorch_ranger import Ranger
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

from utils import (EvalDataset, OUNoise, Prioritized_Buffer, get_beta, 
                   preprocess_data, to_np, hit_metric, dcg_metric)

from classes import *

train_env = Env(train_matrix)
hits, dcgs = [], []
hits_all, dcgs_all = [], []
step, best_step = 0, 0
step, best_step, best_step_all = 0, 0, 0
users = np.random.permutation(appropriate_users)
ou_noise = OUNoise(params['embedding_dim'], decay_period=10)


with open('logs/memory.pickle', 'wb') as f:
    pickle.dump(train_env.memory, f)
    
with open('logs/memory.pickle', 'rb') as f:
    memory = pickle.load(f)

state_repr = State_Repr_Module(user_num, item_num, params['embedding_dim'], params['hidden_dim'])

valid_dataset = EvalDataset(
    np.array(test_data)[np.array(test_data)[:, 0] == 6039], 
    item_num, 
    test_matrix)
valid_loader = td.DataLoader(valid_dataset, batch_size=100, shuffle=False)

full_dataset = EvalDataset(np.array(test_data), item_num, test_matrix)
full_loader = td.DataLoader(full_dataset, batch_size=100, shuffle=False)

def run_evaluation(net, state_representation, training_env_memory, loader=valid_loader):
    hits = []
    dcgs = []
    test_env = Env(test_matrix)
    test_env.memory = training_env_memory.copy()
    user, memory = test_env.reset(int(to_np(next(iter(valid_loader))['user'])[0]))
    for batch in loader:
        action_emb = net(state_repr(user, memory))
        scores, action = net.get_action(
            batch['user'], 
            torch.tensor(test_env.memory[to_np(batch['user']).astype(int), :]), 
            state_representation, 
            action_emb,
            batch['item'].long(), 
            return_scores=True
        )
        user, memory, reward, done = test_env.step(action)

        _, ind = scores[:, 0].topk(10)
        predictions = torch.take(batch['item'], ind).cpu().numpy().tolist()
        actual = batch['item'][0].item()
        hits.append(hit_metric(predictions, actual))
        dcgs.append(dcg_metric(predictions, actual))
        
    return np.mean(hits), np.mean(dcgs)

no_ou_state_repr = State_Repr_Module(user_num, item_num, params['embedding_dim'], params['hidden_dim'])
no_ou_policy_net = Actor_DRR(params['embedding_dim'], params['hidden_dim'])
no_ou_state_repr.load_state_dict(torch.load('logs/final/' + 'best_state_repr.pth'))
no_ou_policy_net.load_state_dict(torch.load('logs/final/' + 'best_policy_net.pth'))
    
# hit, dcg = run_evaluation(no_ou_policy_net, no_ou_state_repr, memory, full_loader)
# print('hit rate: ', hit, 'dcg: ', dcg)

ou_state_repr = State_Repr_Module(user_num, item_num, params['embedding_dim'], params['hidden_dim'])
ou_policy_net = Actor_DRR(params['embedding_dim'], params['hidden_dim'])
ou_state_repr.load_state_dict(torch.load('logs/final/' + 'best_state_repr.pth'))
ou_policy_net.load_state_dict(torch.load('logs/final/' + 'best_policy_net.pth'))

# hit, dcg = run_evaluation(ou_policy_net, ou_state_repr, memory, full_loader)
# print('hit rate: ', hit, 'dcg: ', dcg)

movies = pd.read_csv('data/movies.dat', sep='::', header=None, engine='python', names=['id', 'name', 'genre'],
                     encoding = "ISO-8859-1")
# in the code numeration starts with 0

random_user = np.random.randint(user_num)
movies[movies['id'].isin(np.argwhere(test_matrix[random_user] > 0)[:, 1] + 1)]

def get_random_movies():
    drawn_numbers = []
    for _ in range(3):
        number = random.randint(0, len(movies))
        drawn_numbers.append(number)
    return drawn_numbers

print("\nMovies randomic picked: ")
random_movies = get_random_movies()
for i in random_movies:
    rec = movies.loc[movies['id'] == i]
    print('| Movie name: '+" ".join(rec['name'].to_string().split()[1:])+" | Genre: "+" ".join(rec['genre'].to_string().split()[1:])+" |")
print('\n')
predictions = []
for model, state_representation in zip([ou_policy_net, no_ou_policy_net], [ou_state_repr, no_ou_state_repr]):
    example_env = Env(test_matrix)
    user, memory = example_env.reset(random_user)

    user, memory, reward, _ = example_env.step(torch.tensor([random_movies[0]]))
    user, memory, reward, _ = example_env.step(torch.tensor([random_movies[1]]))
    user, memory, reward, _ = example_env.step(torch.tensor([random_movies[2]]))

    preds = []
    for _ in range(3):
        action_emb = model(state_representation(user, memory))
        action = model.get_action(
            user, 
            torch.tensor(example_env.memory[to_np(user).astype(int), :]), 
            state_representation, 
            action_emb,
            torch.tensor(
                [item for item in example_env.available_items 
                if item not in example_env.viewed_items]
            ).long()
        )
        user, memory, reward, _ = example_env.step(action)
        preds.append(action)

    predictions.append(preds)

technique = ["-->Without Ornstein-Uhlenbeck process: ","-->With Ornstein-Uhlenbeck process: "]
print("Movies recommended by agent: ")
for index,i in enumerate(predictions):
    print()
    print(technique[index])
    for j in i:
        rec = movies.loc[movies['id'] == int(j[0].numpy())]
        print('\t| Movie name: '+" ".join(rec['name'].to_string().split()[1:])+" | Genre: "+" ".join(rec['genre'].to_string().split()[1:])+" |")