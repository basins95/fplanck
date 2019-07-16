import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

K = 1e-6
U = lambda x, y: 0.5*K*(x**2 + y**2)
sim = fokker_planck(temperature=300, drag=drag, extent=[600*nm, 600*nm],
            resolution=10*nm, boundary='periodic', potential=U)

steady = sim.steady_state()

w = 30*nm
x0 = -150*nm
y0 = -150*nm
Pi = lambda x, y: np.exp(-((x - x0)**2 + (y - y0)**2)/w**2)
# def Pi(x, y):
    # p0 = np.zeros_like(x)
    # r = np.sqrt((x - 150*nm)**2 + (y - 150*nm)**2)
    # idx = r < 40*nm
    # p0[idx] = 1
    # return p0

p0 = Pi(*sim.grid)
p0 /= np.sum(p0)

fig, ax = plt.subplots()

Pt = sim.propagate(Pi, 20000e-7)
ax.pcolormesh(*sim.grid/nm, steady)
ax.set_aspect('equal')
# ax.plot(sim.grid[0]/nm, p0, color='red', ls='--', alpha=.3)
# line, = ax.plot(sim.grid[0]/nm, p0, lw=2, color='C3')

# def update(i):
    # time = 3e-6*i
    # Pt = sim.propagate(Pi, time)
    # line.set_ydata(Pt)

    # return [line]

# anim = FuncAnimation(fig, update, frames=range(0, 1000, 3), interval=30)
# ax.set(xlabel='x (nm)', ylabel='normalized PDF')
# ax.margins(x=0)

plt.show()