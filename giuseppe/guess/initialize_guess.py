from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Problem


def process_static_value(input_value: Union[float, ArrayLike], output_len: int):

    input_dim = np.ndim(input_value)
    if input_dim == 0:
        output_array = np.empty((output_len,), dtype=float)
        output_array.fill(input_value)
    elif input_dim == 1:
        output_array = np.asarray(input_value, dtype=float)
    else:
        raise ValueError('Given input_value has more than 1 dimensions')

    if len(output_array) != output_len:
        raise ValueError(f'Cannot match input with shape {len(input_value)} to specified shape {output_len}')

    return output_array


def process_dynamic_value(input_value: Union[float, ArrayLike], output_shape: tuple[int, int]):

    input_dim = np.ndim(input_value)
    if input_dim == 0:
        output_array = np.empty(output_shape, dtype=float)
        output_array.fill(input_value)
    elif input_dim == 1:
        output_array = np.tile(np.reshape(input_value, (output_shape[0], -1)), output_shape[1])
    elif input_dim == 2:
        output_array = np.asarray(input_value, dtype=float)
    else:
        raise ValueError('Given input_value has more than 2 dimensions')

    if output_array.shape != output_shape:
        raise ValueError(f'Cannot match input with shape {np.shape(input_value)} to specified shape {output_shape}')

    return output_array


def initialize_guess(
        prob: Problem, default_value: float = 1.,
        t_span: Union[float, ArrayLike] = 1.,
        x: Optional[ArrayLike] = None,
        p: Optional[ArrayLike] = None,
        u: Optional[ArrayLike] = None,
        lam: Optional[ArrayLike] = None,
        nu0: Optional[ArrayLike] = None,
        nuf: Optional[ArrayLike] = None,
) -> Solution:

    """
    Generate guess where all variables (excluding the independent) are set to a default single constant

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    prob : Problem
        the problem that the guess is for, needed to shape/size of arrays

    default_value : float, default=1.
        the constant all variables, except time (t_span) and constants (problem's default_values), are set to if not
         otherwise specified

    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])

    x : NDArray, Optional
        state values to initialize guess with

    p : NDArray, Optional
        parameter values to initialize guess with

    u : NDArray, Optional
        control values to initialize guess with

    lam : NDArray, Optional
        costate values to initialize guess with

    nu0 : NDArray, Optional
        initial adjoint parameter values to initialize guess with

    nuf : NDArray, Optional
        terminal adjoint parameter values to initialize guess with

    Returns
    -------
    guess : Solution

    """

    data = {'converged': False}

    if isinstance(t_span, float) or isinstance(t_span, int):
        data['t'] = np.asarray([0., t_span], dtype=float)
    else:
        data['t'] = np.asarray(t_span, dtype=float)

    num_t_steps = len(data['t'])

    if hasattr(prob, 'num_states'):
        if x is None:
            x = default_value

        data['x'] = process_dynamic_value(x, (prob.num_states, num_t_steps))

    if hasattr(prob, 'num_parameters'):
        if p is None:
            p = default_value

        data['p'] = process_static_value(p, prob.num_parameters)

    if hasattr(prob, 'default_values'):
        data['k'] = prob.default_values

    if hasattr(prob, 'num_controls'):
        if u is None:
            u = default_value

        data['u'] = process_dynamic_value(u, (prob.num_controls, num_t_steps))

    if hasattr(prob, 'num_costates'):
        if lam is None:
            lam = default_value

        data['lam'] = process_dynamic_value(lam, (prob.num_costates, num_t_steps))

    if hasattr(prob, 'num_initial_adjoints'):
        if nu0 is None:
            nu0 = default_value
        if nuf is None:
            nuf = default_value

        data['nu0'] = process_static_value(nu0, prob.num_initial_adjoints)
        data['nuf'] = process_static_value(nuf, prob.num_terminal_adjoints)

    return Solution(**data)
