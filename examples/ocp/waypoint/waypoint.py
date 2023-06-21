# Example based on Jorris and Cobb's 2009 article in Journal of GC&D
import numpy as np
import giuseppe
import pickle
from pyproj import Proj

# skip JIT compiling for now
giuseppe.utils.compilation.JIT_COMPILE = False

# create problem instance
cav = giuseppe.problems.input.StrInputProb()

cav.set_independent('t')

# Set constants used in EOMs
cav.add_constant('ge', 9.81)
cav.add_constant('r_0', 6.378e6)
cav.add_constant('rho_0', 1.225)
cav.add_constant('h_0', 122000.0) # altitude
cav.add_constant('v_0', 7315.2) # velocity
cav.add_constant('gamma_0', -1.5) # FPA

cav.add_constant('S', )
cav.add_constant('Cl_star', ) # Cl that produces max L/D
cav.add_constant('m', )

# Set quantities used in EOMs
cav.add_expression('B', 'rho_0*r_0*S*Cl_star / (2*m)')

# Waypoint information
# TODO: check if this form of Lat/long is correct
p = Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)
_x_0, _y_0 = p(28.5881, -80.6699)  # N, W
_x_1, _y_1 = p(34.0468, -27.3072)  # N, W
_x_2, _y_2 = p(33.2216, 41.6878)  # N, E
_x_f, _y_f = p(31.6109, 65.7003)  # N, E

cav.add_constant('x_0', _x_0)
cav.add_constant('x_1', _x_1)
cav.add_constant('x_2', _x_2)
cav.add_constant('x_f', _x_f)

cav.add_constant('y_0', _y_0)
cav.add_constant('y_1', _y_1)
cav.add_constant('y_2', _y_2)
cav.add_constant('y_f', _y_f)

# No-fly-zone information
_nfz_x_1, _nfz_y_1 = p(20.2586, -3.4598)  # N, W
cav.add_constant('nfz_x_1', _nfz_x_1)
cav.add_constant('nfz_y_1', _nfz_y_1)
cav.add_constant('nfz_r_1', 177792.0)  # m

_nfz_x_2, _nfz_y_2 = p(55.7308, 58.5615)  # N, E
cav.add_constant('nfz_x_2', _nfz_x_2)
cav.add_constant('nfz_y_2', _nfz_y_2)
cav.add_constant('nfz_r_2', 277800.0)  # m

# Set up equations of motion
cav.add_state('x', 'v*cos(theta)')
cav.add_state('y', 'v*sin(theta)')
cav.add_state('h', 'v*gamma')
cav.add_state('v', '-B*v^2 * exp(-beta*r_0*h) * (1+cl^2) / (2*E_star)')
cav.add_state('gamma', 'B*v*exp(-beta*r_0*h) * cl*cos(sigma) - 1/v + v')
cav.add_state('theta', 'B*v*exp(-beta*r_0*h) * cl*sin(sigma)')

# Introduce the controls
cav.add_control('sigma') # bank angle
cav.add_control('cl') # fraction of Cl_star

# Control constraints
min_sigma = -60*np.pi / 180
max_sigma = 60*np.pi / 180
cav.add_inequality_constraint(
    'control', 'sigma', lower_limit='min_sigma', upper_limit='max_sigma',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_sigma', method='sin'))

min_cl = 0
max_cl = 2
cav.add_inequality_constraint(
    'control', 'cl', lower_limit='min_cl', upper_limit='max_cl',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_cl', method='sin'))

# Terminal Constraints
# TODO: find h_f
cav.add_constraint('terminal', 'x - x_f')
cav.add_constraint('terminal', 'y - y_f')
cav.add_constraint('terminal', 'h - h_f')

cav.add_inequality_constraint(
        'path', '0.5*', lower_limit='min_u', upper_limit='max_u',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin'))