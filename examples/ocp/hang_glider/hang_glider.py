import numpy as np
import giuseppe
import pickle

giuseppe.utils.compilation.JIT_COMPILE = False
glider = giuseppe.problems.input.StrInputProb()

glider.set_independent('t')

# Set constants
rho = 1.13
Cd_0 = 0.034
k = 0.069662
g = 9.80665
m = 100
S = 14
um = 2.5
R = 100

glider.add_constant('rho', rho)
glider.add_constant('Cd_0', Cd_0)
glider.add_constant('k', k)
glider.add_constant('g', g)
glider.add_constant('m', m)
glider.add_constant('S', S)
glider.add_constant('um', um)
glider.add_constant('R', R)

# set quantities used in EOMs
glider.add_expression('Cd', 'Cd_0 + k*(Cl^2)')
glider.add_expression('e', 'x/R')
glider.add_expression('X', '(e-2.5)**2')
glider.add_expression('ua', 'um*(1-X)*exp(-X)')
glider.add_expression('Vy', 'vy - ua')
glider.add_expression('vr', 'sqrt(vx**2 + Vy**2)')
glider.add_expression('L', '0.5*rho*(vr^2)*Cl*S')
glider.add_expression('D', '0.5*rho*(vr^2)*Cd*S')
glider.add_expression('sinN', 'Vy/vr')
glider.add_expression('cosN', 'vx/vr')
glider.add_expression('W', 'm*g')

# set up equations of motion
glider.add_state('x', 'vx')
glider.add_state('y', 'vy')
glider.add_state('vx', '-L*sinN/m - D*cosN/m')
glider.add_state('vy', 'L*cosN/m - D*sinN/m - g')

# set controls
glider.add_control('Cl')

# define cost: maximize final range
glider.set_cost('0', '0', '-x')

# Initial conditions
x_0 = 0.0
y_0 = 1000
vx_0 = 13.23
vy_0 = -1.29

glider.add_constant('x_0', x_0)
glider.add_constant('y_0', y_0)
glider.add_constant('vx_0', vx_0)
glider.add_constant('vy_0', vy_0)

# terminal conditions
y_f = 900
vx_f = 13.23
vy_f = -1.29

glider.add_constant('y_f', y_f)
glider.add_constant('vx_f', vx_f)
glider.add_constant('vy_f', vy_f)

glider.add_constant('eps_Cl', 5e-1)

# define other initial/terminal constraints
glider.add_constraint('initial', 't')
glider.add_constraint('initial', 'x - x_0')
glider.add_constraint('initial', 'y - y_0')
glider.add_constraint('initial', 'vx - vx_0')
glider.add_constraint('initial', 'vy - vy_0')

glider.add_constraint('terminal', 'y - y_f')
glider.add_constraint('terminal', 'vx - vx_f')
glider.add_constraint('terminal', 'vy - vy_f')

def ctrl2reg(cl: np.array) -> np.array:
    return np.arcsin((2 * cl - min_Cl - max_Cl) / (max_Cl - min_Cl))


def reg2ctrl(cl_reg: np.array) -> np.array:
    return 0.5 * ((max_Cl - min_Cl) * np.sin(cl_reg) + max_Cl + min_Cl)


# bound the control
min_Cl = 0
max_Cl = 1.4


glider.add_constant('min_Cl', min_Cl)
glider.add_constant('max_Cl', max_Cl)

glider.add_inequality_constraint(
    'control', 'Cl', lower_limit='min_Cl', upper_limit='max_Cl',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_Cl', method='sin'))

# Create the OCP problem and numeric solver
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_glider = giuseppe.problems.symbolic.SymDual(glider, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_glider, verbose=1, node_buffer=1000)

# Generate guess
Cl_guess = .5
guess = giuseppe.guess_generation.auto_propagate_guess(comp_glider, control=ctrl2reg(np.array([Cl_guess])),
                                                       t_span=.1, initial_states=np.array((x_0, y_0, vx_0, vy_0)))

# guess = giuseppe.guess_generation.InteractiveGuessGenerator(comp_glider, num_solver=num_solver, init_guess=guess).run()

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(100, {'y_f': y_0 - 18}, bisection=True)
cont.add_linear_series(100, {'vx_f': 9, 'vy_f': 1, 'y_f': 995}, bisection=True)
cont.add_linear_series(100, {'y_f': 988, 'vx_f': 9, 'vy_f': .8}, bisection=True)

cont.add_linear_series(100, {'y_f': y_f}, bisection=True)
cont.add_logarithmic_series(200, {'eps_Cl': 1e-6}, bisection=True)

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
