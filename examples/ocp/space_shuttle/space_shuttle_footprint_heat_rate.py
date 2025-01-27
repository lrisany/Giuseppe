import os; os.chdir(os.path.dirname(__file__))  # Set diectory to current location

import pickle

import numpy as np

import giuseppe
from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generators import auto_propagate_guess
from giuseppe.io import InputOCP, SolutionSet
from giuseppe.numeric_solvers.bvp import ScipySolveBVP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP
from giuseppe.problems.ocp import SymOCP
from giuseppe.problems.regularization import PenaltyConstraintHandler
from giuseppe.utils import Timer

giuseppe.utils.compilation.JIT_COMPILE = True

# TODO: WIP Does not fully converge

ocp = InputOCP()

ocp.set_independent('t')

ocp.add_expression('r', 're + h')
ocp.add_expression('g', 'mu / r**2')
ocp.add_expression('rho', 'rho_0 * exp(-h / h_ref)')
ocp.add_expression('dyn_pres', '1 / 2 * rho * v ** 2 ')
ocp.add_expression('lift', 'c_l * s_ref * dyn_pres')
ocp.add_expression('drag', 'c_d * s_ref * dyn_pres')
ocp.add_expression('c_l', 'a_0 + a_1 * alpha_hat')
ocp.add_expression('c_d', 'b_0 + b_1 * alpha_hat + b_2 * alpha_hat**2')
ocp.add_expression('alpha_hat', 'alpha * 180 / pi')
ocp.add_expression('q', 'q_a * q_r')
ocp.add_expression('q_a', 'c_0 + c_1 * alpha_hat + c_2 * alpha_hat**2 + c_3 * alpha_hat**3')
ocp.add_expression('q_r', 'd_0 * sqrt(rho) * (d_1 * v) ** d_2')

ocp.add_state('h', 'v * sin(gamma)')
ocp.add_state('phi', 'v * cos(gamma) * sin(psi) / (r * cos(theta))')
ocp.add_state('theta', 'v * cos(gamma) * cos(psi) / r')
ocp.add_state('v', '-drag / m - g * sin(gamma)')
ocp.add_state('gamma', 'lift * cos(beta) / (m * v) + cos(gamma) * (v / r - g / v)')
ocp.add_state('psi', 'lift * sin(beta)/(m * v * cos(gamma)) + v * cos(gamma) * sin(psi) * sin(theta)/(r * cos(theta))')

ocp.add_control('alpha')
ocp.add_control('beta')

ocp.add_constant('rho_0', 0.002378)
ocp.add_constant('h_ref', 23_800)
ocp.add_constant('re', 20_902_900)
ocp.add_constant('m', 203_000 / 32.174)
ocp.add_constant('mu', 0.14076539e17)

ocp.add_constant('a_0', -0.20704)
ocp.add_constant('a_1', 0.029244)
ocp.add_constant('b_0', 0.07854)
ocp.add_constant('b_1', -0.61592e-2)
ocp.add_constant('b_2', 0.621408e-3)
ocp.add_constant('c_0', 1.0672181)
ocp.add_constant('c_1', -0.19213774e-1)
ocp.add_constant('c_2', 0.21286289e-3)
ocp.add_constant('c_3', -0.10117249e-5)
ocp.add_constant('d_0', 17700)
ocp.add_constant('d_1', 0.0001)
ocp.add_constant('d_2', 3.07)
ocp.add_constant('s_ref', 2690)

ocp.add_constant('xi', 0)

ocp.add_constant('eps_alpha', 1e-5)
ocp.add_constant('alpha_min', -80 / 180 * 3.1419)
ocp.add_constant('alpha_max', 80 / 180 * 3.1419)

ocp.add_constant('eps_q', 0.01)
ocp.add_constant('q_max', 200)

ocp.add_constant('h_0', 260_000)
ocp.add_constant('phi_0', 0)
ocp.add_constant('theta_0', 0)
ocp.add_constant('v_0', 25_600)
ocp.add_constant('gamma_0', -1 / 180 * np.pi)
ocp.add_constant('psi_0', np.pi / 2)

ocp.add_constant('h_f', 80_000)
ocp.add_constant('v_f', 2_500)
ocp.add_constant('gamma_f', -5 / 180 * np.pi)

ocp.set_cost('0', '0', '-phi * cos(xi) - theta  * sin(xi)')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'h - h_0')
ocp.add_constraint('initial', 'phi - phi_0')
ocp.add_constraint('initial', 'theta - theta_0')
ocp.add_constraint('initial', 'v - v_0')
ocp.add_constraint('initial', 'gamma - gamma_0')
ocp.add_constraint('initial', 'psi - psi_0')

ocp.add_constraint('terminal', 'h - h_f')
ocp.add_constraint('terminal', 'v - v_f')
ocp.add_constraint('terminal', 'gamma - gamma_f')

ocp.add_inequality_constraint('path', 'alpha', lower_limit='alpha_min', upper_limit='alpha_max',
                              regularizer=PenaltyConstraintHandler('eps_alpha', method='sec'))
ocp.add_inequality_constraint('path', 'q', upper_limit='q_max',
                              regularizer=PenaltyConstraintHandler('eps_q', method='rat'))

with Timer(prefix='Compilation Time:'):
    sym_ocp = SymOCP(ocp)
    sym_dual = SymDual(sym_ocp)
    sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='differential')
    comp_dual_ocp = CompDualOCP(sym_bvp)
    num_solver = ScipySolveBVP(comp_dual_ocp, bc_tol=1e-8)

guess = auto_propagate_guess(comp_dual_ocp, control=(20/180*3.14159, 0), t_span=100)
with open('guess.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess.k, guess)
print(seed_sol.converged)

with open('seed.data', 'wb') as file:
    pickle.dump(seed_sol, file)

sol_set = SolutionSet(sym_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(100, {'h_f': 200_000, 'v_f': 10_000}, bisection=True)
cont.add_linear_series(50, {'h_f': 80_000, 'v_f': 2_500, 'gamma_f': -5 / 180 * 3.14159}, bisection=True)
# cont.add_linear_series(100, {'alpha_min': -5 / 180 * 3.14159, 'alpha_max': 20 / 180 * 3.14159}, bisection=True)
cont.add_linear_series(90, {'xi': np.pi / 2}, bisection=True)
cont.add_linear_series(200, {'q_max': 100}, bisection=True)
cont.add_linear_series(200, {'q_max': 70}, bisection=True)
# cont.add_logarithmic_series(200, {'eps_q': 1e-7}, bisection=True)
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
