"""
robustoed: robust augmentation for optimal experimental design (OED).

This is a pre-alpha research prototype.
"""

from .types import Design
from .model_sympy import SympyModel
from .grid import make_grid_equidistant
from .optim import wynn_fedorov_d_opt, WynnFedorovOptions, WynnFedorovResult
from .sensitivity import sensitivity_report_d, SensitivityReport
from .sensitivity import sensitivity_report_d_vs_scenario_optimum
from .augment import robust_augment_two_step, RobustAugmentResult

from .screening import screen_uncertain_parameters_d, ParameterScreeningResult
