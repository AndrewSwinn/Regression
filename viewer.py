import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.widgets import CheckButtons

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def covarience(radius, alpha):
    beta = np.sqrt(radius**2 - alpha**2)
    corr = np.array([
            [    1, beta, alpha],
            [ beta,    1,     0],
            [alpha,    0,     1]])

    return corr


samples = 500

corr = covarience(0.95, 0.5)
pca = np.linalg.eig(corr)
data = np.random.multivariate_normal(mean=np.zeros(3), cov=corr, size=samples, check_valid='warn')




# Create 3D figure
fig, (ctl, ax) = plt.subplots(figsize=(15,12), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 4]})

ctl = fig.add_subplot(211, aspect='auto', frame_on=False)
ax  = fig.add_subplot(212, projection='3d', aspect='auto')

ctl.get_xaxis().set_visible(False)
ctl.get_yaxis().set_visible(False)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

x1 = data[:,0]
x2 = data[:,1]
y  = data[:,2]

scatter = ax.scatter(x1, x2, y)

for vector in pca.eigenvectors:
    ax.quiver(0,0,0, vector[0], vector[1], vector[2], color='r')




plt.show()
