import os; os.chdir(os.path.dirname(__file__))  # Set diectory to current location

import giuseppe

goddard = giuseppe.io.InputOCP()

goddard.set_independent('t')

goddard.add_state('h', 'v')
goddard.add_state('v', '(thrust - sigma * v**2 * exp(-h / h_ref))/m - g')
goddard.add_state('m', '-thrust/c')

goddard.add_control('thrust')

goddard.add_constant('max_thrust', 193.044)
goddard.add_constant('g', 32.174)
goddard.add_constant('sigma', 5.49153484923381010e-5)
goddard.add_constant('c', 1580.9425279876559)
goddard.add_constant('h_ref', 23_800)

goddard.add_constant('h_0', 0)
goddard.add_constant('v_0', 0)
goddard.add_constant('m_0', 3)

goddard.add_constant('m_f', 2.95)

goddard.add_constant('eps_thrust', 0.01)

goddard.set_cost('0', '0', '-h')

goddard.add_constraint('initial', 't')
goddard.add_constraint('initial', 'h - h_0')
goddard.add_constraint('initial', 'v - v_0')
goddard.add_constraint('initial', 'm - m_0')

goddard.add_constraint('terminal', 'm - m_f')

goddard.add_inequality_constraint(
        'control', 'thrust', lower_limit='0', upper_limit='max_thrust',
        regularizer=giuseppe.regularization.ControlConstraintHandler('eps_thrust * h_ref', method='atan'))

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(goddard)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp)

guess = giuseppe.guess_generators.auto_linear_guess(comp_dual_ocp)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(sym_bvp, seed_sol)

cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(10, {'m_f': 1})
cont.add_logarithmic_series(20, {'eps_thrust': 1e-6})
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
