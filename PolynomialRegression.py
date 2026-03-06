import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
mpl.rcParams.update({'font.size': 6})


x1      = np.linspace(-5, 5, 11)
x2_base     = x1**2
x2_harmonic = np.sin(x1)
x2          = x2_base - 3 * x2_harmonic



theta = np.linspace(0, 2 * np.pi, 100)
phi   = np.linspace(0, np.pi, 50)

theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

fig = plt.figure(dpi=240)
ax = fig.add_subplot(xticks=[], yticks=[])
#fig.subplots_adjust(left=0.25)

# Make vertical slider to control the polynomial order.
ax_order = fig.add_axes([0.05, 0.10, 0.05, 0.3])

order  = Slider(ax=ax_order, label='Order', valmin=0, valmax=10, valinit=1, valstep=1, orientation="vertical")

#Make box to contain information
axinfo = fig.add_axes([0.05, 0.5, 0.25, 0.45])
axinfo.set_xlim(0,10)
axinfo.set_ylim(0,10)


ax.set_xlim(-5, 5)
ax.set_ylim(-15, 15)



def update(val):

    n = order.val


    ax.plot(x1, x2)


    fig.canvas.draw_idle()


order.on_changed(update)


update(0)


plt.show()