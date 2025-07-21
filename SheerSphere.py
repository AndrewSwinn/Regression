import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
mpl.rcParams.update({'font.size': 6})


theta = np.linspace(0, 2 * np.pi, 100)
phi   = np.linspace(0, np.pi, 50)

theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

fig = plt.figure(dpi=240)
ax = fig.add_subplot(projection='3d', xticks=[], yticks=[], zticks=[])
fig.subplots_adjust(left=0.25)

# Make vertical slider to control the covariences.
axfreq = fig.add_axes([0.05, 0.10, 0.05, 0.3])
bxfreq = fig.add_axes([0.10, 0.10, 0.05, 0.3])
cxfreq = fig.add_axes([0.15, 0.10, 0.05, 0.3])

alpha  = Slider(ax=axfreq, label=r'$\alpha$', valmin=0.0, valmax=1.0, valinit=0.0, valstep=0.05, orientation="vertical")
beta   = Slider(ax=bxfreq, label=r'$\beta$' , valmin=0.0, valmax=1.0, valinit=0.0, valstep=0.05, orientation="vertical")
gamma  = Slider(ax=cxfreq, label=r'$\gamma$', valmin=0.0, valmax=1.0, valinit=0.0, valstep=0.05, orientation="vertical")

#Make box to contain information
axinfo = fig.add_axes([0.05, 0.5, 0.25, 0.45])
axinfo.set_xlim(0,10)
axinfo.set_ylim(0,10)


ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)


def update(val):
    a, b, c = alpha.val, beta.val, gamma.val

    cov = np.array([[1,a,b], [a,1,c],[b,c,1]])
    det = np.linalg.det(cov)

    covX = np.array([[1,a], [a,1]])
    r = np.array([[b], [c]])
    coeff = np.matmul(np.linalg.inv(covX), r)
    R2    = np.matmul(coeff.T, np.matmul(covX, coeff)).item()

    S_Xy =  '1 & ' + str(round(a,2)) + ' & ' + str(round(b,2)) + ' \\\\ ' +  str(round(a,2)) + ' & 1 & ' +      str(round(c,2)) + ' \\\\ ' +  str(round(b,2)) + ' & ' + str(round(c,2)) + '  & 1 '
    S_X  = '1 & ' + str(round(a, 2)) + ' \\\\ ' +  str(round(a,2)) + ' & 1 '
    XTy    = str(round(b,2)) + ' \\\\ ' +  str(round(c,2))
    aV     = str(round(coeff[0].item(),2)) + '\\\\' + str(round(coeff[1].item(),2))



    xT = x + a * y + b * z
    yT = a * x + y + c * z
    zT = b * x + c * y + z

    ax.clear()
    ax.axis('off')

    ax.quiver(0, 0, 0, 2, 0, 0, arrow_length_ratio=0.1, color='black')
    ax.quiver(0, 0, 0, 0, 2, 0, arrow_length_ratio=0.1, color='black')
    ax.quiver(0, 0, 0, 0, 0, 2, arrow_length_ratio=0.1, color='black')

    ax.text(  0,  0, 2.1, s=r'$y$')
    ax.text(2.1,  0,   0, s=r'$x_1$')
    ax.text(  0,2.1,   0, s=r'$x_2$')

    ax.plot_surface(xT, yT, zT, cmap='viridis', alpha=0.2)

    corr = np.array([[1,a,b],[a,1,c], [b,c,1]])
    eig  = np.linalg.eig(corr)
    for i, value in enumerate(eig.eigenvalues):
        L, ux, uy, uz = value, eig.eigenvectors[0][i], eig.eigenvectors[1][i], eig.eigenvectors[2][i]
        ax.quiver(0,0,0, 1*L*ux, 1*L*uy, 1*L*uz, arrow_length_ratio=0.1, color='blue')
        ax.text(1.1*L*ux, 1.1*L*uy, 1.1*L*uz, s=r'$\lambda_' + str(i) + '$')

    axinfo.clear()
    axinfo.set_xlim(0, 10)
    axinfo.set_ylim(0, 10)
    axinfo.axis('off')
    axinfo.text(0.5, 10, r'$\Sigma(X,y)=\begin{pmatrix}' + S_Xy + '\\end{pmatrix}$')
    axinfo.text(0.5, 8, r'$C=\begin{pmatrix}' + S_X + '\\end{pmatrix}$')
    axinfo.text(6.5, 8, r'$r=\begin{pmatrix}' + XTy + '\\end{pmatrix}$')
    axinfo.text(0.5, 6, r'$a=C^{-1}r=\begin{pmatrix}' + aV + '\\end{pmatrix}$')
    axinfo.text(0.5,4, r'$|\Sigma|='+str(round(det,2))+'$')
    axinfo.text(0.5, 2, r'$R^2=a^TCa=' + str(round(R2, 2)) + '$')
    fig.canvas.draw_idle()


alpha.on_changed(update)
beta.on_changed(update)
gamma.on_changed(update)

update(0)


plt.show()