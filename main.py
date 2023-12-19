# !/usr/bin/env python3
# coding=utf-8
import dataclasses
import heapq
import inspect
import itertools
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy
from matplotlib import pyplot
import matplotlib.font_manager
import pandas

numpy.seterr(all='raise')


class Methods:
    @dataclasses.dataclass
    class Arguments:
        thermal_conductivity_sand: float
        percentage_clay: float
        percentage_sand: float
        percentage_silt: float
        porosity_ratio: float
        density_soil_non_si: float
        particle_density: float
        density_soil: float

    class MarkertSoilTypes(Enum):
        SAND = "Sand"
        SILT = "Silt"
        LOAM = "Loam"

    class MarkertSoilStates(Enum):
        Packed = "Packed"
        Unpacked = "Unpacked"
        Both = "Both"

    markert_default_parameters = 1.21, -1.55, 0.02, 0.25, 2.29, 2.12, -1.04, -2.03  # "Sand + Silt + Loam"

    @staticmethod
    def _get_markert_parameters(soil_type: MarkertSoilTypes, soil_state: MarkertSoilStates) -> tuple[float, ...]:
        markert_parameters = {
            (Methods.MarkertSoilTypes.SAND, Methods.MarkertSoilStates.Both):
                (1.02, -1.64, -0.13, 0.25, 1.99, 2.87, -1.32, -2.39),
            (Methods.MarkertSoilTypes.SAND, Methods.MarkertSoilStates.Unpacked):
                (1.51, -3.07, 1.24, 0.24, 1.87, 2.34, -1.34, -1.32),
            (Methods.MarkertSoilTypes.SAND, Methods.MarkertSoilStates.Packed):
                (0.72, -0.74, -0.82, 0.22, 1.55, 2.22, -1.36, -0.95),
            (Methods.MarkertSoilTypes.SILT, Methods.MarkertSoilStates.Both):
                (1.48, -2.15, 0.78, 0.23, 0.00, 0.86, 0.41, 0.20),
            (Methods.MarkertSoilTypes.SILT, Methods.MarkertSoilStates.Unpacked):
                (0.92, -1.08, 0.90, 0.21, 0.14, 1.27, 0.25, -0.33),
            (Methods.MarkertSoilTypes.SILT, Methods.MarkertSoilStates.Packed):
                (1.83, -2.75, 0.12, 0.22, 5.00, 1.32, -1.56, -0.88),
            (Methods.MarkertSoilTypes.LOAM, Methods.MarkertSoilStates.Both):
                (1.64, -2.39, -0.42, 0.28, 3.88, 1.62, -1.10, -2.36),
            (Methods.MarkertSoilTypes.LOAM, Methods.MarkertSoilStates.Unpacked):
                (1.24, -1.55, -0.08, 0.28, 4.26, 1.17, -1.62, -1.19),
            (Methods.MarkertSoilTypes.LOAM, Methods.MarkertSoilStates.Packed):
                (1.79, -2.62, -0.39, 0.25, 3.83, 1.44, -1.11, -2.02),
        }
        return markert_parameters.get((soil_type, soil_state), Methods.markert_default_parameters)

    @staticmethod
    def _interpolate_for_sadhegi(sand_percent: float, silt_percent: float, clay_percent: float) -> dict[str, float]:
        # from Table 4 in Sadeghi et al. (2018)
        soil_samples = {
            "Sand": {
                "sand_percent": 93.0,
                "clay_percent": 5.0,
                "theta_s": 0.395,
                "theta_c": 0.017,
                "lambda_dry": 0.252,
                "lambda_sat": 2.654,
                "t_s": 0.330
            },
            "Loam": {
                "sand_percent": 38.0,
                "clay_percent": 17.0,
                "theta_s": 0.451,
                "theta_c": 0.056,
                "lambda_dry": 0.216,
                "lambda_sat": 1.534,
                "t_s": 0.300
            },
            "Clay": {
                "sand_percent": 23.0,
                "clay_percent": 40.0,
                "theta_s": 0.482,
                "theta_c": 0.132,
                "lambda_dry": 0.198,
                "lambda_sat": 1.310,
                "t_s": 0.242
            },
        }

        def distance(sample: dict) -> float:
            return ((sample["sand_percent"] - sand_percent) ** 2 +
                    ((100 - sample["sand_percent"] - sample["clay_percent"]) - silt_percent) ** 2 +
                    (sample["clay_percent"] - clay_percent) ** 2) ** 0.5

        # Find the three nearest neighbors
        nearest_samples = heapq.nsmallest(3, soil_samples.values(), key=distance)

        # Interpolate values based on the neighbors (for simplicity, we'll average them out)
        lambda_dry = sum(sample["lambda_dry"] for sample in nearest_samples) / 3
        lambda_sat = sum(sample["lambda_sat"] for sample in nearest_samples) / 3
        t_s_val = sum(sample["t_s"] for sample in nearest_samples) / 3
        theta_c_val = sum(sample["theta_c"] for sample in nearest_samples) / 3
        theta_s_val = sum(sample["theta_s"] for sample in nearest_samples) / 3

        return {
            "lambda_dry": lambda_dry,
            "lambda_sat": lambda_sat,
            "t_s": t_s_val,
            "theta_c": theta_c_val,
            "theta_s": theta_s_val
        }

    @staticmethod
    def hu(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        lam_w = .6  # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
        steps = theta / arguments.porosity_ratio
        steps_max = steps.max()
        if 1. < steps_max:
            steps /= steps_max

        particle_density_kg_m_cubed = 2_700

        ke_hu = .9878 + .1811 * numpy.log(steps)
        lam_dry = (
                (0.137 * arguments.density_soil + 64.7) /
                (particle_density_kg_m_cubed - 0.947 * arguments.density_soil)
        )

        lam_s = 3.35

        # Calculate lambda_sat
        lam_sat = lam_w ** arguments.porosity_ratio * lam_s ** (1 - arguments.porosity_ratio)

        ke_hu = numpy.where(steps > .05, ke_hu, 0.)

        lam_hu = lam_dry + ke_hu * (lam_sat - lam_dry)
        return lam_hu

    @staticmethod
    def ewen_and_thomas(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        zeta = -8.9
        steps = theta / arguments.porosity_ratio
        steps_max = steps.max()
        if 1. < steps_max:
            steps /= steps_max
        ke_ma = 1. - numpy.exp(zeta * steps)
        particle_density_kg_m_cubed = 2700
        lam_dry = (.137 * arguments.density_soil + 64.7) / (particle_density_kg_m_cubed - .947 * arguments.density_soil)
        lam_w = 0.57
        volume_fraction_quartz = .5 * arguments.percentage_sand / 100  # Convert percentage to fraction
        lam_q = 7.7
        lam_other = 2

        # Thermal conductivity of soil mineral solids
        lam_s = lam_q ** volume_fraction_quartz * lam_other ** (1 - volume_fraction_quartz)

        lam_sat = lam_w ** arguments.porosity_ratio * lam_s ** (1 - arguments.porosity_ratio)
        lam_ewan_thomas = lam_dry + ke_ma * (lam_sat - lam_dry)
        return lam_ewan_thomas

    @staticmethod
    def brakelmann(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        lam_w = .588  # .57  # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
        lam_b = .0812 * arguments.percentage_sand + .054 * arguments.percentage_silt + .02 * arguments.percentage_clay
        steps = theta / arguments.porosity_ratio
        steps_max = steps.max()
        if 1. < steps_max:
            steps /= steps_max
        lam_brakelmann = (
                lam_w ** arguments.porosity_ratio *
                lam_b ** (1. - arguments.porosity_ratio) *
                numpy.exp(-3.08 * arguments.porosity_ratio * (1. - steps) ** 2)
        )
        return lam_brakelmann

    @staticmethod
    def _markert_all(theta: numpy.ndarray, arguments: Arguments, p: tuple[float, ...]) -> numpy.ndarray:
        lambda_dry = p[0] + p[1] * arguments.porosity_ratio
        alpha = p[2] * arguments.percentage_clay / 100. + p[3]
        _gamma = p[5] * arguments.density_soil_non_si
        _delta = p[6] * arguments.percentage_sand / 100. * arguments.density_soil_non_si
        beta = p[4] * arguments.percentage_sand / 100. + _gamma + _delta + p[7]

        lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))
        return lam_markert

    @staticmethod
    def _markert_soil_type(silt_percentage: float, clay_percentage: float, sand_percentage: float) -> MarkertSoilTypes:
        if silt_percentage + 2 * clay_percentage < 30:
            # TG Sand; S, LS
            return Methods.MarkertSoilTypes.SAND

        # if 50 < silt_percentage and clay_percentage < 27:
        if 50 > sand_percentage and clay_percentage < 27:
            # TG Silt; Si, SiL
            return Methods.MarkertSoilTypes.SILT

        # TG Loam; SL, SCL, SiCL, CL, L
        return Methods.MarkertSoilTypes.LOAM

    @staticmethod
    def markert_unspecific(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        return Methods._markert_all(theta, arguments, p=Methods.markert_default_parameters)

    @staticmethod
    def markert_specific_packed(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        soil_type = Methods._markert_soil_type(arguments.percentage_silt, arguments.percentage_clay, arguments.percentage_sand)
        parameters = Methods._get_markert_parameters(soil_type, Methods.MarkertSoilStates.Packed)
        return Methods._markert_all(theta, arguments, p=parameters)

    @staticmethod
    def markert_specific_unpacked(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        soil_type = Methods._markert_soil_type(arguments.percentage_silt, arguments.percentage_clay, arguments.percentage_sand)
        parameters = Methods._get_markert_parameters(soil_type, Methods.MarkertSoilStates.Unpacked)
        return Methods._markert_all(theta, arguments, p=parameters)

    @staticmethod
    def markert_specific_both(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        soil_type = Methods._markert_soil_type(arguments.percentage_silt, arguments.percentage_clay, arguments.percentage_sand)
        parameters = Methods._get_markert_parameters(soil_type, Methods.MarkertSoilStates.Both)
        return Methods._markert_all(theta, arguments, p=parameters)

    @staticmethod
    def _markert_lu(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        # seite 19, https://www.dropbox.com/s/6iq8z26iahk6s6d/2018-10-25_FAU_TB_Endbericht_Pos.3.1_V3.pdf?dl=0
        lambda_dry = -.56 * arguments.porosity_ratio + .51
        sigma = .67 * arguments.percentage_clay / 100. + .24
        beta = (
                1.97 * arguments.percentage_sand / 100. +
                arguments.density_soil_non_si * 1.87 -
                1.36 * arguments.percentage_sand / 100. -
                .95
        )
        lam_markert_lu = lambda_dry + numpy.exp(beta - theta ** (-sigma))
        return lam_markert_lu

    @staticmethod
    def _devries(theta_l: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function `devries` computes the thermal conductivity for the DeVries model.
        Note that some elements of the DeVries model such as `epsilon_k` are placeholders
        and should be calculated based on given conditions or provided as inputs to the function.
        """

        # TODO: Placeholder. Was not provided in the LaTeX source. Should be calculated based on given conditions or provided.
        epsilon_k = 0.2

        # Constants and table values
        lam_water = 0.57
        lam_vapor_air = 0.035 + 0.298 * (theta_l / arguments.porosity_ratio)

        # Shape factors from Table 2
        g = [0, 0, theta_l / arguments.porosity_ratio, 0.125, 0.125, 0.5]

        # Eq. 3.4
        k = list()
        for i in range(6):
            if i == 1:
                k.append(1)
            else:
                each_lam = lam_vapor_air if i == 2 else 8.8 if i == 3 else 2.0 if i == 4 else 0.25
                part1 = 2 / 3 * (1 / (1 + g[i] * (each_lam / lam_water - 1)))
                part2 = 1 / 3 * (1 / (1 + (each_lam / lam_water - 1) * (1 - 2 * g[i])))
                k.append(part1 + part2)

        # Calculate lam_dry
        top = (
                arguments.porosity_ratio * lam_vapor_air +
                sum(
                    k[i] *
                    theta_l *
                    (lam_vapor_air if i == 2 else 8.8 if i == 3 else 2.0 if i == 4 else 0.25)
                    for i in range(6)
                )
        )

        lam_dry = 1.25 * top / (arguments.porosity_ratio + sum(k[i] * theta_l for i in range(6)))

        # Calculating lam using Eq. 3.1 and Eq. 3.2
        top = sum(
            k[i] *
            (lam_vapor_air if i == 2 else 8.8 if i == 3 else 2.0 if i == 4 else 0.25) *
            theta_l
            for i in range(6)
        )
        bot = sum(k[i] * theta_l for i in range(6))

        lam = numpy.where(
            theta_l <= epsilon_k,
            top / bot,
            lam_dry + (lam_dry - lam_dry) / epsilon_k * theta_l
        )

        return lam

    @staticmethod
    def _sadeghi(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        Computes the thermal conductivity of soil based on its water content using the Sadeghi et al. (2018) model.
        Based on the Percolation-Based Effective-Medium Approximation (P-EMA).

        Parameters:
        - theta: Volumetric water content
        - arguments: Various soil properties required for calculations

        Returns:
        - Thermal conductivity of the soil for given water content
        """

        values = Methods._interpolate_for_sadhegi(
            arguments.percentage_sand, arguments.percentage_silt, arguments.percentage_clay
        )
        lam_sat = values["lambda_sat"]
        lam_dry = values["lambda_dry"]
        t_s = values["t_s"]
        theta_c = values["theta_c"]
        theta_s = values["theta_s"]

        """
        # Constants & Placeholder values:
        # Placeholder: Thermal conductivity at saturated condition. To be determined.
        # Just copied from original paper.
        lam_sat = 2.5

        # Placeholder: Thermal conductivity at dry condition. To be determined.
        # Just copied from original paper.
        lam_dry = .2

        # Placeholder: Volumetric water content of the soil when it's fully saturated.
        # To be determined. Just copied from original paper.
        theta_s = 0.25

        # Critical volumetric water content at which the liquid phase (i.e., water) first
        # forms a continuous path through the medium?! Just copied from original paper.
        theta_c = .0033 * arguments.percentage_clay

        # Placeholder: Scaling factor. To be determined. Just copied from original paper.
        t_s = 1
        
        """

        # Calculating 't' as per Eq. 3.9
        t = 1 / t_s

        # Computing the 'a' coefficients as per Eqs. 3.6, 3.7, and 3.8
        a_1 = (theta_c * lam_sat ** t - (theta_s - theta_c) * lam_dry ** t) / (lam_sat ** t - lam_dry ** t)
        a_2 = (theta_s - theta_c) / (lam_sat ** t - lam_dry ** t)
        a_3 = -theta_c * lam_sat ** t * lam_dry ** t / (lam_sat ** t - lam_dry ** t)

        # Calculating square root term and ensuring non-negative values
        sqrt_term = (a_1 - theta) ** 2 - 4 * a_2 * a_3
        sqrt_term[sqrt_term < 0] = 0

        # Ensuring non-zero denominator
        denominator = 2 * a_2
        if numpy.isscalar(denominator):
            if abs(denominator) < 1e-10:
                denominator = 1e-10

        else:  # If it's an array
            denominator[abs(denominator) < 1e-10] = 1e-10

        # Calculating lambda value as per Eq. 3.5
        lambda_value = ((- (a_1 - theta) + numpy.sqrt(sqrt_term)) / denominator) ** t_s
        lam_sadeghi = lambda_value ** (1 / t)

        return lam_sadeghi

    @staticmethod
    def lu(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the method proposed by Lu et al. (2014) to predict
        soil thermal conductivity based on its water content and other soil properties.
        """

        lam_dry = 0.51 - 0.56 * arguments.porosity_ratio
        delta = 0.67 * arguments.percentage_clay / 100. + 0.24
        beta = (
                1.97 * arguments.percentage_sand / 100. +
                1.87 * arguments.density_soil_non_si -
                1.36 * arguments.density_soil_non_si * arguments.percentage_sand / 100. -
                0.95
        )

        intermediates = beta - theta ** (-delta)
        # Create a mask for values in intermediates that are above -100
        mask = intermediates > -100

        # Initialize lam_lu with zeros
        lam_lu = numpy.zeros_like(intermediates)

        # Perform the operation only for values in intermediates that satisfy the mask
        lam_lu[mask] = numpy.exp(intermediates[mask]) + lam_dry

        return lam_lu

    @staticmethod
    def kersten_johansen_bertermann(theta_volumetric: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Kersten Model (1949) to compute the thermal conductivity of soil based on
        its volumetric water content and other properties. The method uses different equations depending on the
        soil's texture (sandiness).
        """

        theta_gravimetric = 100. * theta_volumetric / arguments.density_soil_non_si
        theta_gravimetric_max = theta_gravimetric.max()
        if 100. < theta_gravimetric_max:
            theta_gravimetric /= theta_gravimetric_max

        if (arguments.percentage_silt + arguments.percentage_clay) < 50.:
            theta_gravimetric = numpy.where(theta_gravimetric < 1., 1., theta_gravimetric)

            lam_kersten = (
                    .1442 *
                    (0.7 * numpy.log10(theta_gravimetric) + 0.4) *
                    10 ** (0.6243 * arguments.density_soil_non_si)
            )  # Eq. 3.18

        else:
            theta_gravimetric = numpy.where(theta_gravimetric < 7., 4., theta_gravimetric)

            lam_kersten = (
                    .1442 *
                    (0.9 * numpy.log10(theta_gravimetric) - 0.2) *
                    10 ** (0.6243 * arguments.density_soil_non_si)
            )  # Eq. 3.19

        return lam_kersten

    @staticmethod
    def johansen(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Johansen Model (1977) to compute the thermal conductivity of soil based on
        its volumetric water content and other properties.
        """
        lam_w = 0.57
        volume_fraction_quartz = 0.5 * arguments.percentage_sand / 100  # Convert percentage to fraction
        lam_q = 7.7
        lam_other = 2 if volume_fraction_quartz >= 0.2 else 3

        # Thermal conductivity of soil mineral solids
        lam_s = lam_q ** volume_fraction_quartz * lam_other ** (1 - volume_fraction_quartz)

        particle_density_kg_m_cubed = 2_700

        # Calculate lambda_dry
        lam_dry = (
                (.137 * arguments.density_soil + 64.7) / (particle_density_kg_m_cubed - .947 * arguments.density_soil)
        )

        # Calculate lambda_sat
        lam_sat = lam_w ** arguments.porosity_ratio * lam_s ** (1 - arguments.porosity_ratio)

        # Saturation degree
        steps = theta / arguments.porosity_ratio
        steps_max = steps.max()
        if 1. < steps_max:
            steps /= steps_max

        if arguments.percentage_clay < 5.:
            ke_johansen = numpy.where(steps > .05, 1 + .7 * numpy.log10(steps), 0.)
        else:
            ke_johansen = numpy.where(steps > .1, 1 + numpy.log10(steps), 0.)

        lam_jo = lam_dry + ke_johansen * (lam_sat - lam_dry)
        return lam_jo

    @staticmethod
    def cote_konrad(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Côté & Konrad model (2005) to compute the thermal conductivity of soil based on
        its volumetric water content and other properties.
        """
        k_values = {
            "Gravel and coarse sand": 4.6,  # raus
            "Medium and fine sand": 3.55,   # > 50% sand
            "Silty and clayey soils": 1.9,  # sonst
            "Organic fibrous soils": 0.6,   # raus
        }

        steps = theta / arguments.porosity_ratio
        steps_max = steps.max()
        if 1. < steps_max:
            steps /= steps_max

        if arguments.percentage_sand > 50:
            soil_material = "Medium and fine sand"
        else:
            soil_material = "Silty and clayey soils"

        kappa = k_values[soil_material]
        k_r = (kappa * steps) / (1. + (kappa - 1.) * steps)

        if soil_material not in k_values:
            raise ValueError(f"Invalid soil material: {soil_material}. Supported values are: {', '.join(k_values.keys())}")

        chi = 0.75
        eta = 1.2
        lam_dry = chi * 10 ** (-eta * arguments.porosity_ratio)

        lam_w = 0.6

        lam_s_table = {
            "silt_n_clay": 2.9,
            "sand": 3.,
        }

        lam_s = lam_s_table["silt_n_clay"] if arguments.percentage_sand < 50 else lam_s_table["sand"]

        # Calculate lambda_sat
        lam_sat = lam_w ** arguments.porosity_ratio * lam_s ** (1 - arguments.porosity_ratio)

        lam_cote_konrad = lam_dry + k_r * (lam_sat - lam_dry)
        return lam_cote_konrad

    @staticmethod
    def yang(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Yang et al. (2005) model to compute the thermal conductivity of soil based on its
        volumetric water content and other properties.
        """
        # particle_density = arguments.particle_density
        particle_density = 2_700  # from Yang et al. (2005)

        k_t = .36
        steps = theta / arguments.porosity_ratio
        steps_max = steps.max()
        if 1. < steps_max:
            steps /= steps_max

        # Assuming sand content represents quartz content
        gravimetric_quartz_content = arguments.percentage_sand / 100
        lam_sat = (
                .5 ** arguments.porosity_ratio *
                (
                        7.7 ** gravimetric_quartz_content * 2 ** (1 - gravimetric_quartz_content)
                ) ** (1 - arguments.porosity_ratio)
        )

        # As used before for dry soil
        lam_dry = (.135 * arguments.density_soil + 64.7) / (particle_density - .947 * arguments.density_soil)

        ke_yang = numpy.exp(k_t * (1 - 1 / steps))
        lam_yang = lam_dry + ke_yang * (lam_sat - lam_dry)
        return lam_yang

    @classmethod
    def get_static_methods(cls) -> list[Callable[..., any]]:
        return [
            value for name, value in inspect.getmembers(cls)
            if isinstance(inspect.getattr_static(cls, name), staticmethod) and not name.startswith("_") and not name.startswith("fix")
        ]


def init_outputs(methods: list[Callable[..., any]]) -> dict[str, list]:
    measurement_output = {
        "Messreihe": list(),
        "#Messungen": list(),
    }
    for each_method in methods:
        measurement_output[f"RMSE {each_method.__name__}"] = list()
        measurement_output[f"BIAS {each_method.__name__}"] = list()

    return measurement_output


def init_scatter_data(methods: list[Callable[..., any]]) -> dict[str, dict[str, list]]:
    scatter_data = {
        each_method.__name__:
            {"model": list(), "data": list(), "is_punctual": list(), "is_in_range": list(), "is_tu": list()}
        for each_method in methods
    }
    return scatter_data


def main() -> None:
    pyplot.rcParams.update(
        {
            "font.family": "Palatino Linotype",
            # "font.size": 32,
        }
    )

    font_manager = matplotlib.font_manager.fontManager
    font_manager.addfont("/home/mark/Downloads/Palatino Linotype.ttf")

    # initialize paths
    input_path = Path("data/")
    input_path.mkdir(parents=True, exist_ok=True)
    output_path = Path("output/")
    output_path.mkdir(parents=True, exist_ok=True)

    methods = Methods.get_static_methods()

    # set output file
    result_overview = output_path / "result_overview.csv"

    # set input file
    measurements_input_file = input_path / "Messdatenbank_FAU_Stand_2023-11-08.xlsx"

    cmap = pyplot.get_cmap("Set1")

    particle_density = 2650  # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html
    thermal_conductivity_quartz = 7.7  # metall?

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    data_density = "low", "high"
    # data_density = "all",
    sand_content = "low", "high", "all"
    water_satura = "low", "high", "all"

    combinations = tuple(
        {
            "data": data,
            "sand": sand,
            "water": water
        }
        for data, sand, water in itertools.product(data_density, sand_content, water_satura)
    )

    for each_combination in combinations:
        # name subset
        sand_subset = each_combination["sand"]
        data_subset = each_combination["data"]
        water_subset = each_combination["water"]
        combination_str = f"data-{data_subset}_sand-{sand_subset}_water-{water_subset}"
        print(combination_str)

        # initialize output paths
        subset_path = output_path / combination_str
        subset_path.mkdir(parents=True, exist_ok=True)
        plot_subset_path = subset_path / "plots"
        plot_subset_path.mkdir(parents=True, exist_ok=True)

        # set output files
        soils_output_file = subset_path / "ergebnisse.xlsx"
        measurements_output_file = subset_path / "model_fit.xlsx"

        scatter_data = init_scatter_data(methods)
        measurement_output = init_outputs(methods)

        dataframes = list()

        for n, row in overview_sheet.iterrows():
            # read info from row
            sheet_index = row.values[0]
            short_name = row.values[1]
            percentage_sand = row.values[2]
            percentage_silt = row.values[3]
            percentage_clay = row.values[4]
            density_soil_non_si = row.values[5]
            soil_type = row.values[6]
            each_density = row.values[9]
            measurement_type = row.values[10]

            # skip if wrong subset

            if data_subset == "low" and each_density != "low":
                continue
            if data_subset == "high" and each_density != "high":
                continue
            if sand_subset == "low" and percentage_sand >= 50:
                continue
            if sand_subset == "high" and percentage_sand < 50:
                continue

            # adapt info
            percentage_sand = 0. if isinstance(percentage_sand, str) else percentage_sand
            percentage_silt = 0. if isinstance(percentage_silt, str) else percentage_silt
            percentage_clay = 0. if isinstance(percentage_clay, str) else percentage_clay

            short_name = short_name if isinstance(short_name, str) else "nan"
            density_soil = density_soil_non_si * 1_000.  # g/cm3 -> kg/m3
            porosity_ratio = 1. - density_soil / particle_density

            steps = numpy.linspace(1, 0, num=50, endpoint=False)[::-1]    # Sättigung
            theta_range = steps * porosity_ratio                                    # Wassergehalt

            theta_quartz = .5 * percentage_sand / 100.
            thermal_conductivity_other = 3. if theta_quartz < .2 else 2.
            thermal_conductivity_sand = thermal_conductivity_quartz ** theta_quartz * thermal_conductivity_other ** (1 - theta_quartz)  # thermal conductivity of stone? soil?

            # bound_lo = min(each_range)
            # bound_hi = max(each_range)

            # read from sheet
            print(f"Messung {sheet_index:d} \t fSand={percentage_sand:.0f}, fSilt={percentage_silt:.0f}, fClay={percentage_clay:.0f}")

            each_sheet = data_measurement_sheets.get(f"{sheet_index}")

            theta_array = each_sheet["θ [cm3/cm3]"].to_numpy()
            is_punctual = "punctual" in measurement_type.lower()
            is_tu = "t/u" in soil_type.lower()
            # is_in_range = (theta_array >= bound_lo) & (bound_hi >= theta_array)

            # filter according to water content
            lambda_array = each_sheet["λ [W/(m∙K)]"].to_numpy()
            if water_subset == "low":
                filter_array = (
                        numpy.isfinite(lambda_array)
                        & (0 < theta_array)
                        & ((theta_array / porosity_ratio) < .5)
                    # & numpy.array([not is_punctual] * len(lambda_array))
                    # & numpy.array([not is_tu] * len(lambda_array))
                    # & is_in_range
                )
            elif water_subset == "high":
                filter_array = (
                        numpy.isfinite(lambda_array)
                        & (0 < theta_array)
                        & ((theta_array / porosity_ratio) >= .5)
                    # & numpy.array([not is_punctual] * len(lambda_array))
                    # & numpy.array([not is_tu] * len(lambda_array))
                    # & is_in_range
                )
            else:
                filter_array = (
                        numpy.isfinite(lambda_array)
                        & (0 < theta_array)
                )

            theta_measurement_volumetric = theta_array[filter_array]
            # theta_measurement = each_sheet["θ [cm3/cm3]"]
            data_measured = lambda_array[filter_array]
            s = theta_measurement_volumetric / porosity_ratio

            if len(theta_measurement_volumetric) < 1 or len(data_measured) < 1:
                print(f"Skipping \"Messung {sheet_index:d}\" due to missing data.")
                continue

            no_measurements = len(data_measured)
            measurement_output["Messreihe"].append(sheet_index)
            measurement_output["#Messungen"].append(no_measurements)

            measurement_type_sequence = ["punctual" in measurement_type.lower()] * no_measurements
            soil_type_sequence = ["t/u" in soil_type.lower()] * no_measurements

            arguments = Methods.Arguments(
                thermal_conductivity_sand=thermal_conductivity_sand,
                percentage_clay=percentage_clay,
                percentage_sand=percentage_sand,
                percentage_silt=percentage_silt,
                porosity_ratio=porosity_ratio,
                density_soil_non_si=density_soil_non_si,
                particle_density=particle_density,
                density_soil=density_soil
            )

            # Methods.fix_arguments(theta_measurement_volumetric, arguments)

            # START measurements
            for each_method in methods:
                model_results = each_method(theta_measurement_volumetric, arguments)
                sse = numpy.sum((data_measured - model_results) ** 2)
                rmse = numpy.sqrt(sse / no_measurements)
                bias = numpy.sum(model_results - data_measured) / numpy.sum(model_results)

                measurement_output[f"RMSE {each_method.__name__}"].append(rmse)
                scatter_data[each_method.__name__]["model"].extend(model_results)
                scatter_data[each_method.__name__]["data"].extend(data_measured)
                scatter_data[each_method.__name__]["is_punctual"].extend(measurement_type_sequence)
                # scatter_data[each_method.__name__]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
                scatter_data[each_method.__name__]["is_tu"].extend(soil_type_sequence)
                measurement_output[f"BIAS {each_method.__name__}"].append(bias)
            # END measurements

            # START ideal values
            soil_output = {f"Feuchte {sheet_index:d}, {short_name:s} [m³%]": theta_range}

            #pyplot.figure()
            #pyplot.title(short_name)
            for i, each_method in enumerate(methods):
                lambda_values = each_method(theta_range, arguments)
                #pyplot.plot(theta_range, lambda_values, c=cmap(i), label=each_method.__name__)
                soil_output[f"{each_method.__name__} {sheet_index:d}, {short_name:s} [W/(mK)]"] = lambda_values

            #pyplot.xlabel("Theta [m³%]")
            #pyplot.ylabel("Lambda [W/(mK)]")
            #pyplot.legend()

            # adds sheet
            soil_df = pandas.DataFrame(soil_output)
            dataframes.append((soil_df, sheet_index, short_name))
            # END ideal values

        # START write model fit
        if 0 < len(dataframes):
            soils_output_handler = pandas.ExcelWriter(soils_output_file)
            for soil_df, sheet_index, short_name in dataframes:
                soil_df.to_excel(soils_output_handler, sheet_name=f"{sheet_index:d} {short_name:s}")
            soils_output_handler.close()

        measurements_output_handler = pandas.ExcelWriter(measurements_output_file)
        measurements_df = pandas.DataFrame(measurement_output)
        measurements_df.to_excel(measurements_output_handler, index=False)
        measurements_output_handler.close()
        # END write model fit

        # START plot measurements against models
        for method, info in scatter_data.items():
            print(f"{method:s}: scatterplotting...")
            pyplot.figure()
            pyplot.xlabel("Messung [W/(mK)]")
            pyplot.ylabel("Modell [W/(mK)]")
            direction = 0.
            measurements = 0
            sum_model = 0.
            sum_delta = 0.
            sum_delta_squared = 0.

            for each_model_value, each_measurement_value in zip(info["model"], info["data"]):
                delta = each_model_value - each_measurement_value
                if not numpy.isnan(delta):
                    direction += delta

                    sum_delta_squared += delta ** 2
                    sum_delta += delta
                    sum_model += each_model_value
                    measurements += 1

            if measurements < 1:
                continue

            rmse = numpy.sqrt(sum_delta_squared / measurements)
            bias = sum_delta / sum_model

            with result_overview.open(mode="a") as result_file:
                result_file.write(f"{combination_str:s};{method:s};{rmse:.3f};{bias:.3f}\n")

            figure = pyplot.figure(figsize=(3, 3))
            axis = figure.add_subplot(111)
            axis.set(
                title=convert_name(method),
                xlabel="Messung [W/(mK)]",
                ylabel="Modell [W/(mK)]",
                xlim=(0, 3),
                ylim=(0, 3),

            )
            axis.plot([0, 3], [0, 3], c="black", linestyle="--", alpha=.3)
            # pyplot.plot([0, 3], [0, 3], c="black", linestyle="--", alpha=.3)

            non_punctual_x = [
                each_x for each_x, each_is_punctual in zip(info["data"], info["is_punctual"])
                if not each_is_punctual
            ]
            non_punctual_y = [
                each_y for each_y, each_is_punctual in zip(info["model"], info["is_punctual"])
                if not each_is_punctual
            ]
            # axis.plot(non_punctual_x, non_punctual_y, c="blue", alpha=.1, linestyle="", marker="o", markersize=.5)
            axis.scatter(non_punctual_x, non_punctual_y, c="blue", alpha=.1, s=.5)

            punctual_x = [
                each_x for each_x, each_is_punctual in zip(info["data"], info["is_punctual"])
                if each_is_punctual
            ]
            punctual_y = [
                each_y for each_y, each_is_punctual in zip(info["model"], info["is_punctual"])
                if each_is_punctual
            ]
            axis.scatter(punctual_x, punctual_y, c="red", alpha=.1, s=.5)
            # pyplot.scatter(punctual_x, punctual_y, c="black", alpha=.8, s=8, linewidths=1, marker="x")

            pyplot.xlim(0, 3)
            pyplot.ylim(0, 3)
            pyplot.savefig((plot_subset_path / f"scatter_{method:s}.pdf").as_posix(), bbox_inches="tight")

            # pyplot.show()
            pyplot.close()
        # END plot measurements against models


def convert_name(name: str) -> str:
    names = {
        'markert_specific_packed': 'Markert et al. (2017) u+p',
        'hu': 'Hu et al. (2001)',
        'markert_specific_unpacked': 'Markert et al. (2017) u',
        'brakelmann': 'Brakelmann (1984)',
        'lu': 'Lu et al. (2014)',
        'markert_specific_both': 'Markert et al. (2017) u+p',
        'cote_konrad': 'Côté & Konrad (2005)',
        'markert_unspecific': 'Markert et al. (2017) unspecific',
        'yang': 'Yang et al. (2005)',
        'ewen_and_thomas': 'Ewen & Thomas (1987)',
        'johansen': 'Johansen (1975)',
        'kersten_johansen_bertermann': 'Kersten (1949)',
    }
    return names[name]


if __name__ == "__main__":
    # todo:
    #  data density: nur noch lo und hi, ohne total
    #  reihenfolge:
    #   kersten
    #   johansen
    #   brakelmann
    #   ewan_and_thomas
    #   hu
    #   cote
    #   yang
    #   lu
    #   markert
    #     unspecific
    #     unpacked und packed
    #     unpacked
    #     packed
    #  datenbank kennzahlen
    #    für
    #      lambda (wärmeleitfähigkeit)
    #      theta (wassergehalt, g/cm^3)
    #      rho (bulk density)
    #      sättigung
    #      clay
    #      silt
    #      sand
    #    kennzahlen:
    #      minimum
    #      1. quantil
    #      median
    #      mean
    #      3. quantil
    #      maximum

    main()
