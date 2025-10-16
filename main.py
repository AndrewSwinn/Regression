import numpy as np
import pandas as pd
import copy
import torch
import math
import sqlite3
from   ucimlrepo import fetch_ucirepo
from   sklearn.model_selection import train_test_split
from   itertools import chain, combinations
from   more_itertools import powerset

def explainer(start, end, data):

    def gamma(players, coalition):
        N = len(players)
        S = len(coalition)
        return math.factorial(S) * math.factorial(N - S - 1) / math.factorial(N)

    def value(coalition):
        vertex = copy.copy(start['position'])
        if len(coalition) > 0:
            for player in coalition:
                vertex[player] = end['position'][player]

        diff = data_space - vertex
        md   = np.sqrt(np.einsum('ij,jk,ik->i', diff, data_cov_inv, diff))

        nearest = targets[np.argmin(md, axis=0)]

        return nearest

    data_space = np.vstack([data['position'], start['position'], end['position']])
    targets    = np.append(data['target'], [start['target'], end['target']])

    data_cov = np.matmul(data_space.T, data_space)/len(data_space)

    data_cov_inv = np.linalg.inv(data_cov)

    player_set = {i for i in range(data_space.shape[1])}
    phi = dict()

    for player in player_set:
        phi[player] = 0.0
        coalitions = player_set - {player}
        for S in powerset(coalitions):

            phi[player] += gamma(player_set, S) * (value(set(S).union({player})) - value(set(S)))




    return start['target'], end['target'], phi



path = 'wine.csv'
wine = pd.read_csv(path)

X = wine.drop(columns=['quality'])
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

players = list(X.columns)
target  = 'quality'

X_cov     = np.array(X_train.cov())
X_cov_inv = np.linalg.inv(X_cov)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test,  y_test  = np.array(X_test),  np.array(y_test)

start_idx, end_idx = 0, 9

explain = {'position': X_train,           'target': y_train}
start   = {'position': X_test[start_idx], 'target': y_test[start_idx]}
end     = {'position': X_test[end_idx],   'target': y_test[end_idx]}

start_val, end_val, phi = explainer(start, end,  explain)

print(start_val, end_val)
phi_T = 0
for player, phi_i in phi.items():
    phi_T += phi_i
    print(players[player], start['position'][player], phi_i, phi_T)