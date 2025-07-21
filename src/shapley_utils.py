import math
import copy
import numpy as np
from   more_itertools import powerset
from   sklearn.model_selection import train_test_split

class dataset:

    def correlation(self):
        positive_semi_def = False
        while not positive_semi_def:
            corr = np.zeros((self.dimensions, self.dimensions), dtype=float)
            for i in range(self.dimensions):
                for j in range(i, self.dimensions):
                    if i == j:
                        corr[i,j] = 1
                    else:
                        corr[i,j] = np.random.rand()
                        corr[j,i] = corr[i,j]
            try:
                test = np.linalg.cholesky(corr)
                positive_semi_def = True
            except np.linalg.LinAlgError:
                pass
        return corr

    def __init__(self, dimensions, samples):
        self.dimensions = dimensions
        self.samples = samples
        self.corr = self.correlation()
        self.data = np.random.multivariate_normal(mean=np.zeros(self.dimensions), cov=self.corr, size=samples)
        self.X, self.y = self.data[:,:-1], self.data[:,-1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.features = {i for i in range(self.dimensions-1)}

    def training(self):
        return self.X_train, self.y_train

    def testing(self):
        return self.X_test, self.y_test




class regression_model:

    def __init__(self, data):

        X, y = data
        self.C = np.matmul(X.T, X) / X.shape[0]
        self.r = np.matmul(X.T, y) / X.shape[0]
        self.a = np.matmul(np.linalg.inv(self.C), self.r)

        self.R2 = np.matmul(self.a.T, np.matmul(self.C, self.a))

    def predict(self, X):
        y_hat = np.matmul(X, self.a)
        return y_hat


class shapley:

    def __init__(self, model, data, grand_coalition):
        self.model     = model
        self.X, self.y = data
        self.grand_coalition = grand_coalition

    def gamma(self, coalition):
        N = len(self.grand_coalition)
        S = len(coalition)
        return math.factorial(S) * math.factorial(N - S - 1) / math.factorial(N)


    def regression_coeff(self):

        def value(coalition):
            X_subset     = self.X[:, list(coalition)]
            model_subset = regression_model((X_subset, self.y))
            return model_subset.R2

        def marginal(player, coalition):
            return value(coalition.union(player)) - value(coalition)

        def phi(player):
            players = self.grand_coalition - player
            return np.sum([self.gamma(coalition) * marginal(player, set(coalition)) for coalition in powerset(players)])

        phi_i = [phi({player}) for player in self.grand_coalition]

        return phi_i

    def shap_add(self, X):

        def masker(X, coalition):
            X_masked = copy.deepcopy(X)
            X_masked[:, list(coalition)] = np.mean(X,axis=0)[list(coalition)]
            return X_masked

        def value(X, coalition):
            return self.model.predict(masker(X, coalition)) - np.mean(self.model.predict(self.X), axis=0)

        def marginal(X, player, coalition):
            return value(X, coalition.union(player)) - value(X, coalition)

        def phi(X,player):
            players = self.grand_coalition - player
            return np.sum([self.gamma(coalition) * marginal(X, player, set(coalition)) for coalition in powerset(players)], axis=0)

        phi_i = np.array([phi(X, {player}) for player in self.grand_coalition]).T

        return phi_i







if __name__ == "__main__":


    print('Testing shapley functions')

    data        = dataset(dimensions=6, samples=1000)
    players     = data.features
    model       = regression_model((data.X_train, data.y_train))
    explainer  = shapley(model, data.training(),  players)

    predictions = model.predict(data.X_test)
    expectation = np.mean(model.predict(data.X_test))

    print('Shapley R2 Explainer')
    print(explainer.regression_coeff())

    print('Shapley SHAP Explainer')

    for idx in range (3):

        print(idx, explainer.shap_add(data.X_test)[idx], sum(explainer.shap_add(data.X_test)[idx]) , predictions[idx]- expectation)