from enum import Enum
from typing import Optional, Union, Callable, Iterable
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from giuseppe.io import Solution
from giuseppe.visualization.components.tk_widgets.data_viewer import TKDataViewer
from giuseppe.utils.typing import EMPTY_ARRAY


class SolutionComponentType(Enum):
    INDEPENDENT = 'independent'
    STATES = 'states'
    COSTATES = 'costates'
    CONTROLS = 'controls'


class DataSelector:
    def __init__(
            self,
            master,
            sol: Solution,
            bindings: Optional[Union[Callable, Iterable[Callable]]] = None,
            comp_type: SolutionComponentType = SolutionComponentType.INDEPENDENT,
            idx: int = 0,
            label: Optional[str] = None
    ):
        self.sol: Solution = sol

        self.comp_type = comp_type
        self.idx = idx

        if label is None:
            self.frame = ttk.Frame(master)
        else:
            self.frame = ttk.LabelFrame(master, text=label)

        self.tk_comp_type = tk.StringVar()
        self.comp_type_mapping: dict = {
            'Independent': SolutionComponentType.INDEPENDENT,
            'State': SolutionComponentType.STATES,
            'Control': SolutionComponentType.CONTROLS,
            'Costates': SolutionComponentType.COSTATES
        }
        self.tk_comp_type.set(self._get_type_key_from_val(self.comp_type))
        self.type_box = ttk.Combobox(self.frame, textvariable=self.tk_comp_type)
        self.type_box['values'] = list(self.comp_type_mapping.keys())
        self.type_box['state'] = 'readonly'
        # self.type_box.grid(row=0, column=0, sticky=NSEW)
        self.type_box.bind('<<ComboboxSelected>>', self._type_selected, add='+')
        self.type_box.pack(side=tk.LEFT)

        self.tk_idx = tk.IntVar()
        self.tk_idx.set(self.idx + 1)
        self.num_elements: int = self._get_num_elements(self.comp_type)
        self.idx_spinbox = ttk.Spinbox(
                self.frame, from_=1, to=self.num_elements, increment=1, textvariable=self.tk_idx,
                command=self._idx_selected)
        self.idx_spinbox.pack(side=tk.RIGHT, pady=5)

        if bindings is None:
            bindings = []
        elif not isinstance(bindings, Iterable):
            bindings = [bindings]
        self.bindings = list(bindings)
        self._set_bindings()

        self.pack = self.frame.pack
        self.grid = self.frame.grid

    def _type_selected(self, event: tk.Event):
        self.comp_type = self.comp_type_mapping[self.tk_comp_type.get()]
        self.num_elements = self._get_num_elements(self.comp_type)
        self.idx_spinbox['to'] = self.num_elements
        self.tk_idx.set(min(self.num_elements, self.idx + 1))
        self._idx_selected()

    def _idx_selected(self):
        self.idx = self.tk_idx.get() - 1

    def _get_data_array(self, comp_type: SolutionComponentType) -> Optional[np.ndarray]:
        if comp_type == SolutionComponentType.INDEPENDENT:
            return self.sol.t
        elif comp_type == SolutionComponentType.STATES:
            return self.sol.x
        elif comp_type == SolutionComponentType.CONTROLS:
            return self.sol.u
        elif comp_type == SolutionComponentType.COSTATES:
            return self.sol.lam
        else:
            raise ValueError('Component type not found')

    def _get_num_elements(self, comp_type: SolutionComponentType) -> int:
        _data_array = self._get_data_array(comp_type)

        if _data_array is None:
            return 0
        elif _data_array.ndim == 1:
            return 1
        else:
            return _data_array.shape[0]

    def get(self) -> np.ndarray:
        _data_array = self._get_data_array(self.comp_type)
        if _data_array is None:
            return EMPTY_ARRAY
        elif _data_array.ndim == 1:
            return _data_array
        else:
            if self.idx <= 0:
                return _data_array[0, :]
            elif self.idx >= _data_array.shape[0]:
                return _data_array[-1, :]
            else:
                return _data_array[self.idx, :]

    def _set_bindings(self):
        for binding in self.bindings:
            self.frame.bind('<FocusIn>', binding, add='+')
            self.frame.bind('<FocusOut>', binding, add='+')

    def _get_type_key_from_val(self, comp_type: Union[SolutionComponentType]) -> str:
        if isinstance(comp_type, str):
            comp_type = comp_type.lower()
        for key, val in self.comp_type_mapping.items():
            if val == comp_type:
                return key
        else:
            return list(self.comp_type_mapping.keys())[0]


class TKSolViewer(TKDataViewer):
    def __init__(
            self,
            master: tk.Tk,
            sol: Solution,
            hor_type: SolutionComponentType = SolutionComponentType.INDEPENDENT,
            hor_idx: int = 0,
            vert_type: SolutionComponentType = SolutionComponentType.STATES,
            vert_idx: int = 0,
    ):
        super().__init__(master, include_navbar=False)
        self.sol: Solution = sol

        self.control_panel = ttk.Frame(self.frame)

        self.hor_data_selector = DataSelector(
                self.control_panel, self.sol, bindings=self._update_event, label='X-Axis Data',
                comp_type=hor_type, idx=hor_idx)
        # self.hor_data_selector.pack(side=tk.LEFT, padx=3, pady=3)
        self.hor_data_selector.grid(row=1, column=0, padx=6, pady=6)

        self.vert_data_selector = DataSelector(
                self.control_panel, self.sol, bindings=self._update_event, label='Y-Axis Data',
                comp_type=vert_type, idx=vert_idx)
        # self.vert_data_selector.pack(side=tk.RIGHT, padx=3, pady=3)
        self.vert_data_selector.grid(row=1, column=1, padx=6, pady=6)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.control_panel, pack_toolbar=False)
        self.toolbar.update()
        # self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar.grid(row=0, column=0, columnspan=3, sticky=tk.NSEW)

        self.control_panel.pack(fill=tk.X)

        self.update()

    def _update_event(self, _):
        self.update()

    def update(self):
        self.hor_data_selector.sol = self.sol
        self.vert_data_selector.sol = self.sol

        h_data = self.hor_data_selector.get()
        v_data = self.vert_data_selector.get()

        if (h_data.shape != v_data.shape) or h_data.ndim != 1:
            h_data, v_data = EMPTY_ARRAY, EMPTY_ARRAY

        self.set_data(h_data, v_data)
