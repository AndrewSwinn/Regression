import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.widgets import CheckButtons

def covarience(radius, alpha):
    beta = np.sqrt(radius**2 - alpha**2)
    corr = np.array([
            [    1, beta, alpha],
            [ beta,    1,     0],
            [alpha,    0,     1]])

    return corr


samples = 500
colours = ['red', 'green', 'black']

corr     = covarience(1, 0.5)
pca      = np.linalg.eig(corr)


data     = np.random.multivariate_normal(mean=np.zeros(3), cov=corr, size=samples, check_valid='warn')
pca_proj = np.matmul(data,pca.eigenvectors)


def get_data(show_pca):
    if not show_pca:
        return data[:,0], data[:,1], data[:,2]
    else:
        return pca_proj[:,0], pca_proj[:,1], pca_proj[:,2]

def get_axis_labels(show_pca):
    if not show_pca:
        return ['x1', 'x2', 'y']
    else:
        return  ['red', 'green', 'black']

show_pca=False

x1, x2, y   = get_data(show_pca)
axis_labels = get_axis_labels(show_pca)

# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=((16,10)))
fig.suptitle('Regression Viewer', fontsize=14)

mosaic = "B..AA;C..AA;D..AA"
axes = fig.subplot_mosaic(mosaic)

axes['A'].remove()
axes['A'] = fig.add_subplot(1, 1, 1, projection='3d')
axes['A'].set_xlabel(axis_labels[0])
axes['A'].set_ylabel(axis_labels[1])
axes['A'].set_zlabel(axis_labels[2])
scatter = axes['A'].scatter(x1, x2, y)
vectors = pca.eigenvectors
for c, value in enumerate(pca.eigenvalues):
    print(value, colours[c])
    axes['A'].quiver(0,0,0, vectors[0][c], vectors[1][c], vectors[2][c], color=colours[c])


axes['B'].set_xlabel(axis_labels[0])
axes['B'].set_ylabel(axis_labels[1])
scatter = axes['B'].scatter(x1, x2)
for c, vector in enumerate(pca.eigenvectors):
    axes['B'].quiver(0,0, vector[0], vector[1], color=colours[c])

axes['C'].set_xlabel(axis_labels[0])
axes['C'].set_ylabel(axis_labels[2])
scatter = axes['C'].scatter(x1, y)
for c, vector in enumerate(pca.eigenvectors):
    axes['C'].quiver(0,0, vector[0], vector[2], color=colours[c])

axes['D'].set_xlabel(axis_labels[1])
axes['D'].set_ylabel(axis_labels[2])
scatter = axes['D'].scatter(x2, y)
for c, vector in enumerate(pca.eigenvectors):
    axes['D'].quiver(0,0, vector[1], vector[2], color=colours[c])


def set_axes():
    axes['A'].set_xlabel(axis_labels[0])
    axes['A'].set_ylabel(axis_labels[1])
    axes['A'].set_zlabel(axis_labels[2])



plt.show()