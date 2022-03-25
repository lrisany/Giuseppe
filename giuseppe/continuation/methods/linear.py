from collections.abc import Hashable, Mapping, Iterator
from copy import copy

import numpy as np

from .abstract import ContinuationSeries
from ..solution_set import SolutionSet
from ...problems.bvp import BVPSol
from ...utils.exceptions import ContinuationError
from ...utils.typing import NPArray


class LinearSeries(ContinuationSeries):
    def __init__(self, num_steps: int, target_mapping: Mapping[Hashable: float], solution_set: SolutionSet):

        super().__init__(solution_set)
        self.num_steps: int = num_steps
        self.target_mapping: Mapping[Hashable: float] = target_mapping
        self._idx_target_pairs: list[tuple[int, float]] =\
            [(idx, tar) for idx, tar in zip(self._get_constant_indices(), target_mapping.values())]
        self._steps: list[NPArray]

    def __iter__(self) -> Iterator[tuple[NPArray, BVPSol]]:
        self._initialize_iter()
        return self._perform_iter()

    def _get_constant_indices(self) -> list[int]:
        indices = []
        for constant_key, target_value in self.target_mapping.items():
            try:
                indices.append(self.solution_set.constant_names.index(constant_key))
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

        return indices

    def _initialize_iter(self):
        current_constants = self.solution_set[-1].k

        target_constants = copy(current_constants)
        for idx, constant_target in self._idx_target_pairs:
            target_constants[idx] = constant_target

        self._steps = list(np.linspace(current_constants, target_constants, self.num_steps + 1))

    def _perform_iter(self):
        for current_constants in self._steps[1:]:
            next_guess = self.solution_set[-1]  # Repeated pulling but judged better than passing arguments
            if not next_guess.converged:
                raise ContinuationError('Last solution did not converge. Continuation cannot continue.')
            yield current_constants, next_guess


class BisectionLinearSeries(LinearSeries):
    def __init__(self, num_steps: int, target_mapping: Mapping[Hashable: float], solution_set: SolutionSet,
                 max_bisections: int = 3):

        LinearSeries.__init__(self, num_steps, target_mapping, solution_set)

        self.max_bisections: int = max_bisections
        self.bisection_counter: int = 0

    @staticmethod
    def _bisect_step(last_constants: NPArray, next_constants: NPArray) -> NPArray:
        return (next_constants + last_constants)/2

    def _perform_iter(self):
        self._steps.reverse()
        last_constants = self._steps.pop()

        while self._steps:
            next_constants = self._steps[-1]
            last_sol = self.solution_set[-1]

            if last_sol.converged:
                if self.bisection_counter > 0:
                    self.bisection_counter -= 1
                last_constants = self._steps.pop()

                yield next_constants, last_sol

            elif self.bisection_counter < self.max_bisections:
                # TODO Investigate logging options
                print(f'Last continuation {next_constants} did not converge. Bisecting next step')
                self.bisection_counter += 1
                self._steps.append(self._bisect_step(last_constants, next_constants))
                self.solution_set.damned_sols.append(self.solution_set.pop())

                yield self._steps[-1], self.solution_set[-1]

            else:
                raise ContinuationError('Bisection limit exceeded!')