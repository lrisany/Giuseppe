from abc import abstractmethod
from collections.abc import Iterable, MutableSequence, Hashable
from typing import Union, overload

from ..problems.bvp import SymBVP, BVPSol
from ..utils.mixins import Picky


# TODO: add annotations to solution set
class SolutionSet(MutableSequence, Picky):

    SUPPORTED_INPUTS = Union[SymBVP]

    def __init__(self, problem: SymBVP, seed_solution: BVPSol):
        Picky.__init__(self, problem)

        self.problem: SymBVP = problem
        self.seed_solution: BVPSol = seed_solution
        self.solutions: list[BVPSol] = [seed_solution]
        self.continuation_slices: list[slice] = []
        self.damned_sols: list[BVPSol] = []

        # Annotations
        self.constant_names: tuple[Hashable, ...] = tuple(str(constant) for constant in self.problem.constants)

    def insert(self, index: int, solution: BVPSol) -> None:
        self.solutions.insert(index, solution)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> BVPSol: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[BVPSol]: ...

    def __getitem__(self, i: int) -> BVPSol:
        return self.solutions.__getitem__(i)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: BVPSol) -> None: ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[BVPSol]) -> None: ...

    def __setitem__(self, i: int, o: BVPSol) -> None:
        self.__setitem__(i, o)

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i: int) -> None:
        self.solutions.__delitem__(i)

    def __len__(self) -> int:
        return self.solutions.__len__()