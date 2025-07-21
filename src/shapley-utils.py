import numpy as np

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




if __name__ == "__main__":


    print('Testing shapley functions')

    data = dataset(6,1000)
    print(data.X.shape, data.y.shape)