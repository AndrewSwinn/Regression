import numpy as np
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



class model:

    def __init__(self, dataset):

        self.dataset = dataset
        self.C = np.matmul(dataset.X_train.T, dataset.X_train) / dataset.X_train.shape[0]
        self.r = np.matmul(dataset.X_train.T, dataset.y_train).T / dataset.X_train.shape[0]
        self.a = np.matmul(np.linalg.inv(self.C), self.r)

        self.R2 = np.matmul(self.a.T, np.matmul(self.C, self.a))



if __name__ == "__main__":


    print('Testing shapley functions')

    model = model(dataset(6,1000))
    print(model.R2)