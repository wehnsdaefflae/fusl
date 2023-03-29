import dataclasses
from typing import Callable

import numpy


@dataclasses.dataclass
class Composition:
    percentage_sand: float
    percentage_silt: float
    percentage_clay: float


@dataclasses.dataclass
class Available:
    steps: numpy.ndarray
    data: numpy.ndarray
    theta_measurement: numpy.ndarray
    lambda_measurement: numpy.ndarray
    particle_density: float = 2650
    thermal_conductivity_water: float = .57    # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
    thermal_conductivity_quartz: float = 7.7   # metall?

    @property
    def theta_range(self) -> numpy.ndarray:
        return self.steps * self.porosity_percentage

    @property
    def step_measurement(self) -> numpy.ndarray:
        return self.theta_measurement / self.porosity_percentage

    @property
    def short_name(self) -> str:
        return self.data["short_name"]

    @property
    def composition_percentages(self) -> Composition:
        return Composition(self.data["percentage_sand"], self.data["percentage_silt"], self.data["percentage_clay"])

    @property
    def density_soil(self, si: bool = True) -> float:
        return self.data["density_soil_non_si"] * float(si) * 1000.

    @property
    def porosity_percentage(self) -> float:
        return 1. - self.density_soil / self.particle_density

    @property
    def theta_quartz(self) -> float:
        return .5 * self.composition_percentages.percentage_sand / 100.

    """
    thermal_conductivity_other = 3. if theta_quartz < .2 else 2.
    thermal_conductivity_sand = thermal_conductivity_quartz ** theta_quartz * thermal_conductivity_other ** (
                1 - theta_quartz)  # thermal conductivity of stone? soil?
    """

class Model:
    @staticmethod
    def output(available: Available) -> numpy.ndarray:
        raise NotImplementedError()
