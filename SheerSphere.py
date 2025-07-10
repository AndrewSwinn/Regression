import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

theta = np.linspace(0, 2 * np.pi, 100)
phi   = np.linspace(0, np.pi, 50)

theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

fig = plt.figure(dpi=240)
ax = fig.add_subplot(projection='3d', xticks=[], yticks=[], zticks=[])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.20, 0.65, 0.03])
bxfreq = fig.add_axes([0.25, 0.15, 0.65, 0.03])
cxfreq = fig.add_axes([0.25, 0.10, 0.65, 0.03])
alpha = Slider(ax=axfreq, label=r'$\alpha$', valmin=0.0, valmax=1.0, valinit=0.0)
beta  = Slider(ax=bxfreq, label=r'$\beta$' , valmin=0.0, valmax=1.0, valinit=0.0)
gamma = Slider(ax=cxfreq, label=r'$\gamma$', valmin=0.0, valmax=1.0, valinit=0.0)


ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

#ax.plot_surface(xT, yT, zT, cmap='viridis', alpha=0.8)

ax.quiver(0, 0, 0, 2, 0, 0, arrow_length_ratio=0.1, color='black')
ax.quiver(0, 0, 0, 0, 2, 0, arrow_length_ratio=0.1, color='black')
ax.quiver(0, 0, 0, 0, 0, 2, arrow_length_ratio=0.1, color='black')

def update(val):
    a, b, c = alpha.val, beta.val, gamma.val

    xT = x + a * y + b * z
    yT = a * x + y + c * z
    zT = b * x + c * y + z

    ax.clear()
    ax.plot_surface(xT, yT, zT, cmap='viridis', alpha=0.8)


    corr = np.array([[1,a,b],[a,1,c], [b,c,1]])
    eig  = np.linalg.eig(corr)
    for i, value in enumerate(eig.eigenvalues):
        L, ux, uy, uz = value, eig.eigenvectors[0][i], eig.eigenvectors[1][i], eig.eigenvectors[2][i]
        ax.quiver(1*L*ux, 1*L*uy, 1*L*uz, 2*L*ux, 2*L*uy, 2*L*uz, arrow_length_ratio=0.1, color='black')

    fig.canvas.draw_idle()


alpha.on_changed(update)
beta.on_changed(update)
gamma.on_changed(update)

update(0)


plt.show()