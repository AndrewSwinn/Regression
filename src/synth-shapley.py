import os
import math
import copy
import pandas as pd
from itertools import chain,  combinations

from lark.utils import OrderedSet
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def Shapley(X,y, model_base, value_function):

    def get_value(S, results):
        if len(S) == 0: return 0
        for coalition, value in results:
            if S == set(coalition):
                return value

    def gamma(players, coalition):
        N = len(players)
        S = len(coalition)
        return math.factorial(S) * math.factorial(N - S - 1) / math.factorial(N)

    players = OrderedSet(X.columns)

    #Step 1: Train and evaluate model for all possible coalitions
    results = []
    for coalition_tuple in powerset(players):
        coalition = set(coalition_tuple)
        if len(coalition) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X[list(coalition)], y, test_size=0.2, random_state=42)
            model = copy.deepcopy(model_base)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            value  = value_function(y_test, y_pred)
            results.append([coalition, value])
        else:
            results.append([coalition, 0])


    #Step 2: For each feature evaluate marginal contributions
    phi_i = dict()
    for i in players:
        phi = 0
        players_minus = players - {i}
        for coalition_minus in powerset(players_minus):
            coalition_minus = set(coalition_minus)
            coalition       = coalition_minus.union({i})
            marginal        = get_value(coalition, results) - get_value(coalition_minus, results)
            phi += gamma(players, coalition_minus) * marginal
        phi_i[i] = phi

    return phi_i

def Experiments(X, y, models, value_function ):
    results = pd.DataFrame(index=list(X.columns), columns=models.keys())
    for name, model in models.items():
        phi_i = Shapley(X, y, model, value_function)
        print(name)
        for feature, phi in phi_i.items():
            print(feature,',', phi)
            results.at[feature, name] = phi
        print(results)
    return results


if __name__ == "__main__":

    #Prepare data
    wine = pd.read_csv(os.path.join(os.pardir, 'wine.csv'))
    y = wine['quality']
    X = wine.drop(columns=['quality'])

    models = {'Multilayer Perceptron': MLPRegressor(loss='squared_error', hidden_layer_sizes=(20, 20), max_iter=10000),
              'SupportVectorMachine' : SVR(),
              'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
              'RandomForestRegressor': RandomForestRegressor(random_state=42),
              'LinearRegression'     : LinearRegression()}

    results = Experiments(X, y, models, r2_score )

    print(results)


