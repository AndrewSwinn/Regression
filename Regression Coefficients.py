import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(dpi=240)
ax = fig.add_subplot(projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, 1, 100)
p = np.linspace(0, 0.5*np.pi, 100)
R, P = np.meshgrid(r, p)
#Z = ((R**2 - 1)**2)

# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)
Z = Y**2 / (1 - X**2)

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)

# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1)
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$R^2$')
plt.savefig('r2-alpha-beta')
plt.show()