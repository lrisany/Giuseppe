import os; os.chdir(os.path.dirname(__file__))  # Set diectory to file location

import numpy as np

from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generators import initialize_guess_w_default_value
from giuseppe.io import InputBVP, SolutionSet
from giuseppe.numeric_solvers.bvp.scipy import ScipySolveBVP
from giuseppe.problems.bvp import SymBVP, CompBVP
from giuseppe.utils import Timer

sturm_liouville = InputBVP()

sturm_liouville.set_independent('x')

sturm_liouville.add_state('y', 'yp')
sturm_liouville.add_state('yp', '-k**2 * y')

sturm_liouville.add_parameter('k')

sturm_liouville.add_constant('x_0', 0.)
sturm_liouville.add_constant('x_f', 1.)
sturm_liouville.add_constant('y_0', 0.)
sturm_liouville.add_constant('y_f', 0.)
sturm_liouville.add_constant('a', 1.)

sturm_liouville.add_constraint('initial', 'x - x_0')
sturm_liouville.add_constraint('initial', 'y - y_0')
sturm_liouville.add_constraint('initial', 'yp - a * k')

sturm_liouville.add_constraint('terminal', 'x - x_f')
sturm_liouville.add_constraint('terminal', 'y - y_f')

with Timer(prefix='Compilation Time:'):
    sym_bvp = SymBVP(sturm_liouville)
    comp_bvp = CompBVP(sym_bvp)
    num_solver = ScipySolveBVP(comp_bvp)

guess = initialize_guess_w_default_value(comp_bvp, t_span=np.linspace(0, 1, 3))
seed_sol = num_solver.solve(guess.k, guess)

sol_set = SolutionSet(sym_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(10, {'a': 100})
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
