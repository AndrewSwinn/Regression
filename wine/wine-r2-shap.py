import numpy as np
import pandas as pd
import os
import math
import copy
from   more_itertools import powerset
from   operator import itemgetter
from   ucimlrepo import fetch_ucirepo
from   sklearn.model_selection import train_test_split

project_root = '/mnt/c/Users/aswin/GitHub/Regression'

# Load and split the data
#wine = pd.read_csv(os.path.join(project_root,'wine','wine.csv'))
wine = pd.read_csv('C:\\Users\\ams90\\PycharmProjects\\Regression\\wine\\wine.csv')


pred_dict = {0:'fixed acidity',        1:'volatile acidity', 2:'citric acid', 3:'residual sugar',  4:'chlorides', 5:'free sulfur dioxide',
             6:'total sulfur dioxide', 7:'density',          8:'pH',          9:'sulphates',      10:'alcohol'}

pred_set  = set(pred_dict.values())

def value(S):
    X = np.array(wine[list(S)])
    y = np.array(wine[['quality']])
    X, y = (X - X.mean(axis=0)) / X.std(axis=0), (y - y.mean()) / y.std()
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    R2 = np.matmul(w.T, np.matmul(np.matmul(X.T, X), w)) / len(y)
    return R2

def marginal(j, S):
    return value(S.union(j)) - value(S)

def gamma(N,S):
    return math.factorial(len(S)) * math.factorial(len(N) - len(S) - 1) / math.factorial(len(N))

def phi(j, N):
    players = N - j
    return np.sum([gamma(N, S) * marginal(j, set(S)) for S in powerset(players)])

phi_players = {player: round(phi({player}, pred_set),5) for player in pred_set}

sorted_items = sorted(phi_players.items(), key=lambda kv: (kv[1], kv[0]))



R2=0
for player, value in phi_players.items():
    R2 += value
    print(player, ',', value)
print('Total:', R2)






#wine_quality = fetch_ucirepo(id=186)
#wine_subset  = wine_quality['data']['original'][wine_quality['data']['original']['color'] == 'white']
#X = np.array(wine_subset[['fixed_acidity', 'volatile_acidity', 'citric_acid','residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']])
#y = np.array(wine_subset['quality'])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)