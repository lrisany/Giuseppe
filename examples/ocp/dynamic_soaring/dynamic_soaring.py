import numpy as np
import giuseppe
import pickle

giuseppe.utils.compilation.JIT_COMPILE=False
prob = giuseppe.problems.input.StrInputProb()

prob.set_independent('t')

# auxiliary information
W_0 = 0
rho = 0.002378
Cd_0 = 0.00873
k = 0.045
A = 45.09703
g = 32.2
m = 5.6
rho_bar = 60

# set up equations of motion
prob.add_state('x', 'v * cos(gam) * sin(psi) + Wx')
prob.add_state('y', 'v * cos(gam) * cos(psi)')
prob.add_state('z', 'v * sin(gam)')
prob.add_state('v', '-D / m - g*sin(gam) - Wx_dot * cos(gam)*sin(psi)')
prob.add_state('gam', '(L * cos(sig) - m * g*cos(gam) + m * Wx_dot * sin(gam) * sin(psi)) / (m*v)')
prob.add_state('psi', '(L * sin(sig) - m * Wx_dot * cos(psi)) / (m * v * cos(gam))')

# quantities used in EOMs
prob.add_expression('L', '0.5*rho*(v^2)*Cl*A')
prob.add_expression('D', '0.5*rho*(v^2)*Cd*A')
prob.add_expression('Cd', 'Cd_0 + k*(Cl^2)')
prob.add_expression('Wx', 'beta * z + W_0')
prob.add_expression('Wx_dot', 'beta*v*sin(gam)')

prob.add_expression('beta', 'sqrt(rho*(g^2)*A / (rho_bar*2*m*g))')

prob.add_constant('rho_bar', rho_bar)

# set controls
prob.add_control('Cl')
prob.add_control('sig')

prob.add_parameter('v_bound')
prob.add_parameter('gam_bound')
prob.add_parameter('psi_bound')


# define cost: minimize gradient slope
prob.set_cost('0', '0', 't')

# Initial conditions
x_0 = 0.0
y_0 = 0.0
z_0 = 0.0

prob.add_constant('x_0', x_0)
prob.add_constant('y_0', y_0)
prob.add_constant('z_0', z_0)

prob.add_constant('m', m)
prob.add_constant('W_0', W_0)
prob.add_constant('A', A)
prob.add_constant('rho', rho)
prob.add_constant('Cd_0', Cd_0)
prob.add_constant('k', k)
prob.add_constant('g', g)

# terminal conditions
x_f = 0.0
y_f = 0.0
z_f = 0.0

delta_v = 0.0
delta_gam = 0.0
delta_psi = 0.0

prob.add_constant('x_f', x_f)
prob.add_constant('y_f', y_f)
prob.add_constant('z_f', z_f)

prob.add_constant('delta_v', delta_v)
prob.add_constant('delta_gam', delta_gam)
prob.add_constant('delta_psi', delta_psi)

# Set max/min limits on variables
min_tf = 1.0
max_tf = 30.0
min_x = -1000.0
max_x = 1000.0
min_y = -1000.0
max_y = 1000.0
min_z = 0.0
max_z = 1000.0
min_v = 10.0
max_v = 350.0
min_gam = -75 * np.pi / 180
max_gam = 75 * np.pi / 180
min_psi = -3 * np.pi
max_psi = np.pi / 2

min_beta = 0.005
max_beta = 0.15
min_Cl = -0.5
max_Cl = 0.15
min_sig = -75 * np.pi / 180
max_sig = 75 * np.pi / 180

prob.add_constant('min_tf', min_tf)
prob.add_constant('max_tf', max_tf)
prob.add_constant('min_x', min_x)
prob.add_constant('max_x', max_x)
prob.add_constant('min_y', min_y)
prob.add_constant('max_y', max_y)
prob.add_constant('min_z', min_z)
prob.add_constant('max_z', max_z)
prob.add_constant('min_v', min_v)
prob.add_constant('max_v', max_v)
prob.add_constant('min_gam', min_gam)
prob.add_constant('max_gam', max_gam)
prob.add_constant('min_psi', min_psi)
prob.add_constant('max_psi', max_psi)

prob.add_constant('min_beta', min_beta)
prob.add_constant('max_beta', max_beta)
prob.add_constant('min_Cl', min_Cl)
prob.add_constant('max_Cl', max_Cl)
prob.add_constant('min_sig', min_sig)
prob.add_constant('max_sig', max_sig)

prob.add_constant('eps_Cl', 1e-6)
prob.add_constant('eps_sig', 1e-6)
prob.add_constant('eps_beta', 1e-6)


# define other initial/terminal constraints
prob.add_constraint('initial', 't')
prob.add_constraint('initial', 'x - x_0')
prob.add_constraint('initial', 'y - y_0')
prob.add_constraint('initial', 'z - z_0')
prob.add_constraint('initial', 'v - v_bound')
prob.add_constraint('initial', 'gam - gam_bound')
prob.add_constraint('initial', 'psi - psi_bound')

prob.add_constraint('terminal', 'x - x_f')
prob.add_constraint('terminal', 'y - y_f')
prob.add_constraint('terminal', 'z - z_f')
prob.add_constraint('terminal', 'v - v_bound + delta_v')
prob.add_constraint('terminal', 'gam - gam_bound + delta_gam')
prob.add_constraint('terminal', 'psi - psi_bound + delta_psi')


# bound the controls
prob.add_inequality_constraint(
    'control', 'Cl', lower_limit='min_Cl', upper_limit='max_Cl',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_Cl', method='sin'))

prob.add_inequality_constraint(
    'control', 'sig', lower_limit='min_sig', upper_limit='max_sig',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_sig', method='sin'))


# Create the OCP problem and numeric solver
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_prob = giuseppe.problems.symbolic.SymDual(prob, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_prob, verbose=1, node_buffer=1000)

Cl_guess = 0
sig_guess = 0

# Generate guess
guess = giuseppe.guess_generation.auto_propagate_guess(comp_prob, control= np.array((Cl_guess, sig_guess)),
                                                       p= np.array((100,-.005, 0)), t_span=10, initial_states=np.array((x_0, y_0, z_0, 100,-.005, 0)))
guess = giuseppe.guess_generation.InteractiveGuessGenerator(comp_prob, num_solver=num_solver, init_guess=guess).run()

# control= np.array((Cl_guess, sig_guess))
with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(100, {'x_f': x_f, 'y_f': y_f})
cont.add_linear_series(100, {'z_f': z_f})
cont.add_linear_series(100, {'gam_f': gam_f, 'psi_f': psi_f})
cont.add_logarithmic_series(200, {'eps_Cl': 1e-6})
cont.add_logarithmic_series(200, {'eps_sig': 1e-6})

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
