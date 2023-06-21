# Example based on Pager and Rao's 2022 Scitech paper
import numpy as np
import giuseppe
import pickle
from pyproj import Proj

# skip JIT compiling for now
giuseppe.utils.compilation.JIT_COMPILE = False

# create problem instance
craft = giuseppe.problems.input.StrInputProb()

craft.set_independent('t')

# Set constants used in EOMs, inertially symmetric rigid body
craft.add_constant('a', 0)
craft.add_constant('w3_0', -0.3)

# Set quantities used in EOMs

# Set up equations of motion
craft.add_state('w_1', 'a*w3_0*w_2 + u_1')
craft.add_state('w_2', '-a*w3_0*w_1 + u_2')
craft.add_state('x_1', 'w3_0*x_2 + w_2*x_1*x_2 + 0.5*w_1*(1 + x_1**2 - x_2**2)')
craft.add_state('x_2', 'w3_0*x_1 + w_1*x_1*x_2 + 0.5*w_2*(1 + x_2**2 - x_1**2)')

# Smoothing parameters
craft.add_constant('eps_u1', 1e-1)
craft.add_constant('eps_u2', 1e-1)

# Introduce the controls
craft.add_control('u_1')
craft.add_control('u_2')

# Control constraints
u1_min = -1
u1_max = 1
u2_min = -1
u2_max = 1

craft.add_constant('u1_min', u1_min)
craft.add_constant('u1_max', u1_max)
craft.add_constant('u2_min', u2_min)
craft.add_constant('u2_max', u2_max)

craft.add_inequality_constraint(
    'control', 'u_1', lower_limit='u1_min', upper_limit='u1_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u1', method='sin'))

craft.add_inequality_constraint(
    'control', 'u_2', lower_limit='u2_min', upper_limit='u2_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u2', method='sin'))

# Initial Constraints
w1_0 = 0
w2_0 = 0
x1_0 = 0
x2_0 = 0

craft.add_constant('w1_0', w1_0)
craft.add_constant('w2_0', w2_0)
craft.add_constant('x1_0', x1_0)
craft.add_constant('x2_0', x2_0)

craft.add_constraint('initial', 't')
craft.add_constraint('initial', 'w_1 - w1_0')
craft.add_constraint('initial', 'w_2 - w2_0')
craft.add_constraint('initial', 'x_1 - x1_0')
craft.add_constraint('initial', 'x_2 - x2_0')

# Final Constraints
w1_f = 1
w2_f = 2
craft.add_constant('w1_f', w1_f)
craft.add_constant('w2_f', w2_f)

craft.add_constraint('terminal', 'w_1 - w1_f')
craft.add_constraint('terminal', 'w_2 - w2_f')

# Cost function
craft.set_cost('0', '0', 't')

# Create the complete OCP with solver
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_craft = giuseppe.problems.symbolic.SymDual(craft, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_craft, verbose=0, max_nodes=0, node_buffer=10)

# Fcns to regularize the controls, used in guess generation
u_min = -1  # assuming the same bounds
u_max = 1


def ctrl2reg(u: np.array) -> np.array:
    return np.arcsin((2 * u - u_min - u_max) / (u_max - u_min))


def reg2ctrl(u_reg: np.array) -> np.array:
    return 0.5 * ((u_max - u_min) * np.sin(u_reg) + u_max + u_min)


# Generate guess
u1_guess = 1  # guesses from original paper's solution
u2_guess = 1
guess = giuseppe.guess_generation.auto_propagate_guess(comp_craft, control=ctrl2reg(np.array([u1_guess, u2_guess])),
                                                       t_span=0.1, initial_states=np.array((w1_0, w2_0, x1_0, x2_0)))
guess = giuseppe.guess_generation.InteractiveGuessGenerator(comp_craft, num_solver=num_solver, init_guess=guess).run()

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(100, {'w1_f': w1_f, 'w2_f': w2_f})
cont.add_logarithmic_series(200, {'eps_u1': 1e-6})
cont.add_logarithmic_series(200, {'eps_u2': 1e-6})

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
