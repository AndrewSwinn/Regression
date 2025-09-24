import os
import math
import pandas as pd
from itertools import chain,  combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def Shapley(X,y, model_type, value_function):
    def value(S, results):
        if len(S) == 0: return 0
        for coalition, value in results:
            if S == set(coalition):
                return value

    def gamma(players, coalition):
        N = len(players)
        S = len(coalition)
        return math.factorial(S) * math.factorial(N - S - 1) / math.factorial(N)

    players = set(X.columns)

    #Step 1: Train and evaluate model for all possible coalitions
    results = []
    for coalition_tuple in powerset(players):
        coalition = set(coalition_tuple)
        if len(coalition) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X[list(coalition)], y, test_size=0.2, random_state=42)
            model = model_type(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = value_function(y_test, y_pred)
            results.append([coalition, 1 - mae])
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
            marginal        = value(coalition, results) - value(coalition_minus, results)
            phi += gamma(players, coalition) * marginal
        phi_i[i] = phi

    return phi_i

if __name__ == "__main__":

    #Prepare data
    wine = pd.read_csv(os.path.join(os.pardir, 'wine.csv'))
    y = wine['quality']
    X = wine.drop(columns=['quality'])

    Shapley = Shapley(X, y, RandomForestRegressor, mean_absolute_error)


    print(Shapley)