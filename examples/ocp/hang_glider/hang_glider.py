import numpy as np
import giuseppe
import pickle

giuseppe.utils.compilation.JIT_COMPILE = False
prob = giuseppe.problems.input.StrInputProb()

prob.set_independent('t')

# Set constants
rho = 1.13
Cd_0 = 0.034
k = 0.069662
g = 9.80665
m = 100
S = 14
um = 2.5
R = 100

prob.add_constant('rho', rho)
prob.add_constant('Cd_0', Cd_0)
prob.add_constant('k', k)
prob.add_constant('g', g)
prob.add_constant('m', m)
prob.add_constant('S', S)
prob.add_constant('um', um)
prob.add_constant('R', R)

# set quantities used in EOMs
prob.add_expression('Cd', 'Cd_0 + k*(Cl^2)')
prob.add_expression('e', 'x/R')
prob.add_expression('X', '(e-2.5)^2')
prob.add_expression('ua', 'um*(1-X)*exp(-X)')
prob.add_expression('Vy', 'vy - ua')
prob.add_expression('vr', 'sqrt(vx^2 + Vy^2)')
prob.add_expression('L', '0.5*rho*(vr^2)*Cl*S')
prob.add_expression('D', '0.5*rho*(vr^2)*Cd*S')
prob.add_expression('sinN', 'Vy/vr')
prob.add_expression('cosN', 'vx/vr')
prob.add_expression('W', 'm*g')

# set up equations of motion
prob.add_state('x', 'vx')
prob.add_state('y', 'vy')
prob.add_state('vx', '-L*sinN - D*cosN')
prob.add_state('vy', 'L*cosN - D*sinN - g')

# set controls
prob.add_control('Cl')

# define cost: maximize final range
prob.set_cost('0', '0', '-x')

# Initial conditions
x_0 = 0.0
y_0 = 1000
vx_0 = 13.23
vy_0 = -1.29

prob.add_constant('x_0', x_0)
prob.add_constant('y_0', y_0)
prob.add_constant('vx_0', vx_0)
prob.add_constant('vy_0', vy_0)

# terminal conditions
y_f = 900
vx_f = 13.23
vy_f = -1.29

prob.add_constant('y_f', y_f)
prob.add_constant('vx_f', vx_f)
prob.add_constant('vy_f', vy_f)

prob.add_constant('eps_Cl', 1e-6)

# define other initial/terminal constraints
prob.add_constraint('initial', 't')
prob.add_constraint('initial', 'x - x_0')
prob.add_constraint('initial', 'y - y_0')
prob.add_constraint('initial', 'vx - vx_0')
prob.add_constraint('initial', 'vy - vy_0')

prob.add_constraint('terminal', 'y - y_f')
prob.add_constraint('terminal', 'vx - vx_f')
prob.add_constraint('terminal', 'vy - vy_f')

# bound the control
min_Cl = 0
max_Cl = 1.41

prob.add_constant('min_Cl', min_Cl)
prob.add_constant('max_Cl', max_Cl)

prob.add_inequality_constraint(
    'control', 'Cl', lower_limit='min_Cl', upper_limit='max_Cl',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_Cl', method='sin'))

# Create the OCP problem and numeric solver
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_prob = giuseppe.problems.symbolic.SymDual(prob, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_prob, verbose=1, node_buffer=1000)

# Generate guess
Cl_guess = .7
guess = giuseppe.guess_generation.auto_propagate_guess(comp_prob, control= np.array(Cl_guess),
                                                       t_span=10, initial_states=np.array((x_0, y_0, vx_0, vy_0)))

guess = giuseppe.guess_generation.InteractiveGuessGenerator(comp_prob, num_solver=num_solver, init_guess=guess).run()

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(1000, {'y_f': y_f, 'vy_f': vy_f})
cont.add_linear_series(1000, {'vx_f': vx_f})
cont.add_logarithmic_series(200, {'eps_Cl': 1e-6})

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
