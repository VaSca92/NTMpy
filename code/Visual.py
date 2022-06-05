# =============================================================================
#   TITLE: NTMpy visual
# -----------------------------------------------------------------------------
#        Authors: Valentino Scalera, Lukas Alber
#        Version: 2.0
#   Dependencies: numpy, matplotlib, bspline, tqdm
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as movie

# ==============================================================================
#
# ==============================================================================
def spaceVStime(x,t,phi):
    plt.contourf(x,t,phi,50,cmap = 'plasma')
    plt.colorbar( orientation='vertical', shrink=0.8)
    plt.xlabel('x-Space')
    plt.ylabel('time')
    plt.title(r'$\phi(t,x)$')
    plt.show()

# ==============================================================================
#
# =============================================================================
def single_point(t,phi,i):
    plt.plot(t,phi[:,i])
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Evolution of Amplitude at one point in space')
    plt.grid()
    plt.show()

# ==============================================================================
#
# ==============================================================================
def animated(x,t,phi,speed = 2):
    fig, ax = plt.subplots()
    minAx = np.min(phi) - .1*np.min(phi)*np.sign(np.min(phi));
    maxAx = np.max(phi) + .1*np.max(phi)*np.sign(np.max(phi));
    ax.set_xlim(0, x[-1]); ax.set_ylim( minAx, maxAx)
    plt.xlabel('Depth of Material'); plt.ylabel('Temperature')
    line, = ax.plot([], [], 'r', animated=True)
    text  = ax.text(0.02, 0.95, "", transform = ax.transAxes, family='monospace')

    def update(frame):
        line.set_data(x,phi[speed*frame])
        text.set_text("time = " + "{:10.4f}".format(t[speed*frame]) + " s")
        return line, text

    frames = round(min(phi.shape[0], t.shape[0])/speed)
    ani = movie(fig, update, frames = frames, blit = True , interval = 15, repeat = False)
    plt.grid()
    plt.show()

# ==============================================================================
#
# ==============================================================================
def compare(x,t,phi1,phi2, speed = 1):
    fig, ax = plt.subplots()
    minAx1 = np.min(phi1); maxAx1 = np.max(phi1);
    minAx2 = np.min(phi2); maxAx2 = np.max(phi2);
    minAx = min([minAx1, minAx2]); maxAx = max([maxAx1, maxAx2]);
    diff = maxAx - minAx;
    ax.set_xlim(0, x[-1]); ax.set_ylim( minAx - .1*diff, maxAx + .1*diff)
    plt.xlabel('Depth of Material'); plt.ylabel('Temperature')
    line1, = ax.plot([], [], 'r', animated=True)
    line2, = ax.plot([], [], 'b', animated=True)
    text = ax.text(0.02, 0.95, "", transform = ax.transAxes)

    def update(frame):
        line1.set_data(x,phi1[frame*speed]); line2.set_data(x,phi2[frame*speed])
        text.set_text("time = " + "{:10.4}".format(t[frame*speed]) + " s")
        return line1, line2, text

    ani = movie(fig, update, blit=True, interval = 15)
    plt.grid()
    plt.show()

# ==============================================================================
#
# ==============================================================================
def surface(x, t, phi):
    x, t = np.meshgrid(x[::2], t[::10])
    fig = plt.figure(); ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,t,phi[::10,::2],cmap = 'plasma')
    fig.colorbar(surf,shrink=0.7, aspect=5)
    plt.xlabel('x-Space'); plt.ylabel('time'); plt.title(r'$\phi(x,t)$')
    plt.show()

# ==============================================================================
#
# ==============================================================================
def average(x, t, phi):
    dx = np.diff(x); wx = (np.insert(dx, 0, 0) + np.append(dx, 0) )/2
    if len(phi[0].shape) > 1:
        for i in range(len(phi)):
            phi_ave = np.average(phi[i], axis = 1, weights = wx)
            plt.plot(t, phi_ave);
    else:
        phi_ave = np.average(phi, axis = 1, weights = wx)
        plt.plot(t, phi_ave);
    plt.grid(); plt.show();
