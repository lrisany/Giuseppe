import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA = 2

if DATA == 0:
    with open('guess.data', 'rb') as f:
        sol = pickle.load(f)
elif DATA == 1:
    with open('seed_sol.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

# Un-regularize the control
eps_u = sol.k[3] # with first index as 1
u_min = sol.k[5]
u_max = sol.k[6]

u = 0.5 * ((u_max - u_min) * np.sin(sol.u) + u_max + u_min)

# PLOT STATES
ylabs = (r'$\omega_1$', r'$\omega_2$', r'$x_1$', r'$x_2$')
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state)

fig_states.tight_layout()

# PLOT Control
ylabs = (r'$u_1$', r'$u_2$')
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(u)):
    axes_u.append(fig_u.add_subplot(1, 2, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl)

fig_u.tight_layout()

plt.show()
