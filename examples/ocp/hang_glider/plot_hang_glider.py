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

eps_Cl = sol.k[15]
Cl_min = sol.k[16]
Cl_max = sol.k[17]
Cl = 0.5 * ((Cl_max - Cl_min) * np.sin(sol.u) + Cl_max + Cl_min)

dJ_dt = np.sum(Cl, 0)

# PLOT STATES
ylabs = (r'$x$', r'$y$', r'$v_x$', r'$v_y$')
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

# PLOT Cl
ylabs = (r'$Cl$')
fig_Cl = plt.figure()
axes_Cl = []

for idx, ctrl in enumerate(list(Cl)):
    axes_Cl.append(fig_Cl.add_subplot(1, 1, idx + 1))
    ax = axes_Cl[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('$C_l$')
    ax.plot(sol.t, ctrl)
    ax.set_title(f'Total Cost = {sol.cost}')

fig_Cl.tight_layout()

# PLOT COSTATES
ylabs = (r'$x$', r'$y$', r'$v_x$', r'$v_y$')
fig_lam = plt.figure()
axes_lam = []

for idx, lam in enumerate(list(sol.lam)):
    axes_lam.append(fig_lam.add_subplot(2, 2, idx + 1))
    ax = axes_lam[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, lam)

fig_lam.tight_layout()




plt.show()
