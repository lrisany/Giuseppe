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

# Set constants used in EOMs

# Set quantities used in EOMs

# Set up equations of motion
craft.add_state('w1_dot', 'a*w3_0*w_2 + u_1')
craft.add_state('w2_dot', '-a*w3_0*w_1 + u_2')
craft.add_state('w3_0*x_2 + w_2*x_1*x_2 + 0.5*w_1*(1 + x_1**2 - x_2**2')
craft.add_state('w3_0*x_1 + w_1*x_1*x_2 + 0.5*w_2*(1 + x_2**2 - x_1**2')

# Introduce the controls
craft.add_control('u_1')
craft.add_control('u_2')