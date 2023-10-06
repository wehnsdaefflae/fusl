# !/usr/bin/env python3
# coding=utf-8
import dataclasses
import inspect
from pathlib import Path
from typing import Callable

import numpy
from matplotlib import pyplot
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

    @staticmethod
    def _calculate_lam_dry(theta_l: numpy.ndarray, arguments: Arguments, lam_vapor_air: numpy.ndarray, k: list) -> numpy.ndarray:
        """
        Calculates the thermal conductivity of dry soil, denoted as `lam_dry`, based on the provided parameters.

        Thermal Conductivity (Λ) and Shape Factor (g_i) of Soil Constituents (with Φ = porosity):

        | Soil Constituent | i | Λ (W·m^{-1}·K^{-1})    | g_i                                |
        |------------------|---|------------------------|------------------------------------|
        | Water            | 1 | 0.57                   | -                                  |
        | Vapor and Air    | 2 | Λ_vapor + Λ_air        | 0.035 + 0.298 * (Θ_L / Φ)          |
        | Quartz           | 3 | 8.8                    | 0.125                              |
        | Other Minerals   | 4 | 2.0                    | 0.125                              |
        | Organic Matter   | 5 | 0.25                   | 0.5                                |

        Parameters:
            theta_l (numpy.ndarray): Volumetric water content array.
            arguments (Arguments): A class containing the necessary parameters.
            lam_vapor_air (numpy.ndarray): The thermal conductivity of vapor-air mixture, calculated as a function of `theta_l`.
            k (list): Shape factors derived from the table above.

        Returns:
            numpy.ndarray: The calculated `lam_dry` values corresponding to the input `theta_l`.

        """
        lam_dry = 1.25 * (arguments.porosity_ratio * lam_vapor_air + sum(
            k[i] * theta_l * (lam_vapor_air if i == 2 else 8.8 if i == 3 else 2.0 if i == 4 else 0.25) for i in range(6))) / (
                          arguments.porosity_ratio + sum(k[i] * theta_l for i in range(6)))
        return lam_dry

    @staticmethod
    def hu(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        \section{The model of Hu et al. (2001)}
        The Kersten number $\mathrm{Ke}$ represents a relative thermal conductivity which is defined by
        $$
        K e=\frac{\lambda-\lambda_{d r y}}{\lambda_{\text {sat }}-\lambda_{d r y}}
        $$
        where $\lambda_{d r y}$ and $\lambda_{\text {sat }}$ denote the thermal conductivity of dry and saturated soil, respectively, yielding
        $$
        \lambda=\lambda_{d r y}+\operatorname{Ke}(S)\left(\lambda_{s a t}-\lambda_{d r y}\right)  % Eq. 3.17
        $$
        where $S$ is the saturation degree [22].
        Johansen uses the basic Eq. (3.17) of all Kersten approximations [28]. He and coauthors estimated the thermal conductivity of natural dry soils by
        $$
        \lambda_{d r y}=\frac{0.135 \rho_b+64.7}{\rho_s-0.947 \rho_b} % Eq. 3.20
        $$
        For crushed rocks he proposes
        $$
        \lambda_{d r y}=0.039 \phi^{-2.2}  % Eq. 3.21
        $$
        with $\Phi$ porosity, approximated by $\Phi=1-\rho_{\mathrm{b}} / \rho_{\mathrm{s}}$.
        The model of Hu et al. [30], resembles that of Johansen except the Kersten coefficient which is given by
        $$
        K e=0.9878+0.1811 \ln (S)
        $$
        To estimate the boundaries, Eqs. 3.20 and 3.21 are used with slightly changed constants which are $\lambda_s=3.35 \mathrm{~W} \cdot \mathrm{m}^{-1}{ }^{\circ} \mathrm{K}^{-1}, \lambda_w=0.6 \mathrm{~W} \cdot \mathrm{m}^{-1}{ }^{\circ} \mathrm{K}^{-1}$ ), and $\lambda_{\text {air }}=0.0246$ $\mathrm{W} \cdot \mathrm{m}^{-1}{ }^{\circ} \mathrm{K} \cdot{ }^{-1}$
        """
        lam_w = thermal_conductivity_water = .57  # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
        steps = theta / arguments.porosity_ratio
        ke_hu = .9878 + .1811 * numpy.log(steps)
        # lambda_s = 3.35
        # lambda_w = 0.6
        # lambda_air = 0.0246
        lam_dry = (0.135 * arguments.density_soil + 64.7) / (arguments.particle_density - 0.947 * arguments.density_soil)
        ke_hu[ke_hu == numpy.inf] = lam_dry
        ke_hu[ke_hu == -numpy.inf] = lam_dry
        lam_sat = lam_w ** arguments.porosity_ratio * arguments.thermal_conductivity_sand ** (1 - arguments.porosity_ratio)
        lam_hu = lam_dry + ke_hu * (lam_sat - lam_dry)
        return lam_hu

    @staticmethod
    def markle(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        \section{The model of Markle et al. (2006)}
                The Kersten number $\mathrm{Ke}$ represents a relative thermal conductivity which is defined by
        $$
        K e=\frac{\lambda-\lambda_{d r y}}{\lambda_{\text {sat }}-\lambda_{d r y}}
        $$
        where $\lambda_{d r y}$ and $\lambda_{\text {sat }}$ denote the thermal conductivity of dry and saturated soil, respectively, yielding
        $$
        \lambda=\lambda_{d r y}+\operatorname{Ke}(S)\left(\lambda_{s a t}-\lambda_{d r y}\right)  % Eq. 3.17
        $$
        where $S$ is the saturation degree [22].
        Johansen uses the basic Eq. (3.17) of all Kersten approximations [28]. He and coauthors estimated the thermal conductivity of natural dry soils by
        $$
        \lambda_{d r y}=\frac{0.135 \rho_b+64.7}{\rho_s-0.947 \rho_b} % Eq. 3.20
        $$
        For crushed rocks he proposes
        $$
        \lambda_{d r y}=0.039 \phi^{-2.2}  % Eq. 3.21
        $$
        with $\Phi$ porosity, approximated by $\Phi=1-\rho_{\mathrm{b}} / \rho_{\mathrm{s}}$.
        Following Ewen and Thomas [32] and Markle et al. [33], who proposed an exponential function to obtain Ke by
        $$
        K e=1-\exp (-\zeta S)
        $$
        where $\zeta$ represents a fitting parameter the value of which should be $\zeta=8.9$. To obtain thermal conductivity, again Eqs. 3.17, 3.20, and 3.22 are used. For further details concerning the history of the method, readers are referred to He et al. [12].
        """
        lam_w = thermal_conductivity_water = .57  # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
        zeta = 8.9
        steps = theta / arguments.porosity_ratio
        ke_ma = 1. - numpy.exp(-zeta * steps)
        lam_dry = (.135 * arguments.density_soil + 64.7) / (arguments.particle_density - .947 * arguments.density_soil)
        lam_sat = lam_w ** arguments.porosity_ratio * arguments.thermal_conductivity_sand ** (1 - arguments.porosity_ratio)
        lam_markle = lam_dry + ke_ma * (lam_sat - lam_dry)
        return lam_markle

    @staticmethod
    def brakelmann(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        \section{The Model of Brakelmann}
        Brakelmann developed an empirical formula to estimate soil thermal conductivity based upon bulk density and water content [7]. His method has been used successfully for the planning of buried power cables:
        $$
        \lambda=\lambda_w^{\Phi} \lambda_b^{(1-\Phi)} \exp \left(-3.08 \Phi(1-S)^2\right)
        $$
        with $\lambda_b$ thermal conductivity of soil mineral solids, approximated by: $\lambda_{\mathrm{b}}=0.0812 * \mathrm{sa}$ nd $\%+0.054 *$ silt $\%+0.02 *$ clay $\%, \rho_{\mathrm{p}}$ particle density, approximated by
        $$
        \rho_p=0.0263 * \text { sand } \%+0.0265 * \text { silt } \%+0.028 * \text { clay } \%
        $$
        Saturation degree, $S=\theta / \Phi$.
        """
        lam_w = thermal_conductivity_water = .57  # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
        lam_b = .0812 * arguments.percentage_sand + .054 * arguments.percentage_silt + .02 * arguments.percentage_clay
        # rho_p = 0.0263 * f_sand + 0.0265 * f_silt + 0.028 * f_clay
        steps = theta / arguments.porosity_ratio
        lam_brakelmann = lam_w ** arguments.porosity_ratio * lam_b ** (1. - arguments.porosity_ratio) * numpy.exp(-3.08 * arguments.porosity_ratio * (1. - steps) ** 2)
        return lam_brakelmann

    @staticmethod
    def markert_all(theta: numpy.ndarray, arguments: Arguments, p: tuple[float, ...] | None = None) -> numpy.ndarray:
        if p is None:
            sand_silt_loam = +1.21, -1.55, +0.02, +0.25, +2.29, +2.12, -1.04, -2.03
            p = sand_silt_loam

        lambda_dry = p[0] + p[1] * arguments.porosity_ratio
        alpha = p[2] * arguments.percentage_clay / 100. + p[3]
        beta = p[4] * arguments.percentage_sand / 100. + p[5] * arguments.density_soil_non_si + p[6] * arguments.percentage_sand / 100. * arguments.density_soil_non_si + \
               p[7]
        lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))
        return lam_markert

    @staticmethod
    def _markert_all(theta: numpy.ndarray, arguments: Arguments, packed: bool = True) -> numpy.ndarray:
        # p table: 1323, texture groups: 1320, https://www.dropbox.com/s/y6hm5m6necbzkpr/Soil%20Science%20Soc%20of%20Amer%20J%20-%202017%20-%20Markert%20-.pdf?dl=0

        tg_sand_u = +1.51, -3.07, +1.24, +0.24, +1.87, +2.34, -1.34, -1.32
        tg_sand_p = +0.72, -0.74, -0.82, +0.22, +1.55, +2.22, -1.36, -0.95

        tg_silt_u = +0.92, -1.08, +0.90, +0.21, +0.14, +1.27, +0.25, -0.33
        tg_silt_p = +1.83, -2.75, +0.12, +0.22, +5.00, +1.32, -1.56, -0.88

        tg_loam_u = +1.24, -1.55, +0.08, +0.28, +4.26, +1.17, -1.62, -1.19
        tg_loam_p = +1.79, -2.62, -0.39, +0.25, +3.83, +1.44, -1.11, -2.02

        if arguments.percentage_silt + 2 * arguments.percentage_clay < 30:
            # TG Sand; S, LS
            p = tg_sand_p if packed else tg_sand_u

        elif 50 < arguments.percentage_silt and arguments.percentage_clay < 27:
            # TG Silt; Si, SiL
            p = tg_silt_p if packed else tg_silt_u

        else:
            # TG Loam; SL, SCL, SiCL, CL, L
            p = tg_loam_p if packed else tg_loam_u

        return Methods.markert_all(theta, arguments, p=p)

    @staticmethod
    def markert_specific_packed(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        return Methods._markert_all(theta, arguments, packed=True)

    @staticmethod
    def markert_specific_unpacked(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        return Methods._markert_all(theta, arguments, packed=False)

    @staticmethod
    def markert_lu(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        # seite 19, https://www.dropbox.com/s/6iq8z26iahk6s6d/2018-10-25_FAU_TB_Endbericht_Pos.3.1_V3.pdf?dl=0
        lambda_dry = -.56 * arguments.porosity_ratio + .51
        sigma = .67 * arguments.percentage_clay / 100. + .24
        beta = 1.97 * arguments.percentage_sand / 100. + arguments.density_soil_non_si * 1.87 - 1.36 * arguments.percentage_sand / 100. - .95
        lam_markert_lu = lambda_dry + numpy.exp(beta - theta ** (-sigma))
        return lam_markert_lu

    @staticmethod
    def devries(theta_l: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
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
        k = []
        for i in range(6):
            if i == 1:
                k.append(1)
            else:
                each_lam = lam_vapor_air if i == 2 else 8.8 if i == 3 else 2.0 if i == 4 else 0.25
                part1 = 2 / 3 * (1 / (1 + g[i] * (each_lam / lam_water - 1)))
                part2 = 1 / 3 * (1 / (1 + (each_lam / lam_water - 1) * (1 - 2 * g[i])))
                k.append(part1 + part2)

        # Calculate lam_dry
        lam_dry = Methods._calculate_lam_dry(theta_l, arguments, lam_vapor_air, k)

        # Calculating lam using Eq. 3.1 and Eq. 3.2
        lam = numpy.where(
            theta_l <= epsilon_k,
            (sum(k[i] * (lam_vapor_air if i == 2 else 8.8 if i == 3 else 2.0 if i == 4 else 0.25) * theta_l for i in range(6))) / (
                sum(k[i] * theta_l for i in range(6))),
            lam_dry + (lam_dry - lam_dry) / epsilon_k * theta_l
        )

        return lam

    @staticmethod
    def sadeghi(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        Computes the thermal conductivity of soil based on its water content using the Sadeghi et al. (2018) model.
        Based on the Percolation-Based Effective-Medium Approximation (P-EMA).

        Parameters:
        - theta: Volumetric water content
        - arguments: Various soil properties required for calculations

        Returns:
        - Thermal conductivity of the soil for given water content
        """

        # Constants & Placeholder values:
        # Placeholder: Thermal conductivity at saturated condition. To be determined.
        lam_sat = arguments.thermal_conductivity_sand
        # Placeholder: Critical volumetric water content value. To be determined.
        theta_c = 0.2
        # Placeholder: Volumetric water content of the soil when it's fully saturated. To be determined.
        theta_s = 0.5
        # Placeholder: The formula for lam_dry. Assumed, as not provided in the LaTeX.
        lam_dry = (.135 * arguments.density_soil + 64.7) / (arguments.particle_density - .947 * arguments.density_soil)

        # Calculating 't' as per Eq. 3.9
        t_s = (lam_sat ** (1 / theta_s))
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
        delta = 0.67 * arguments.percentage_clay + 0.24
        beta = 1.97 * arguments.percentage_sand + 1.87 * arguments.density_soil - 1.36 * arguments.density_soil * arguments.percentage_sand - 0.95

        # Conditional computation to avoid underflow
        lam_lu_values_for_small_theta = lam_dry
        threshold = -709  # adjust as needed

        mask = (beta - theta ** (-delta)) > threshold
        exp_value = numpy.zeros_like(theta)
        exp_value[mask] = numpy.exp(beta - theta[mask] ** (-delta))

        lam_lu_values_for_large_theta = exp_value + lam_dry

        lam_lu = numpy.where(theta < 1e-5, lam_lu_values_for_small_theta, lam_lu_values_for_large_theta)

        return lam_lu

    @staticmethod
    def kersten(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Kersten Model (1949) to compute the thermal conductivity of soil based on
        its volumetric water content and other properties. The method uses different equations depending on the
        soil's texture (sandiness).
        """
        if (arguments.percentage_silt + arguments.percentage_clay) > .5:
            lam_kersten = 0.1442 * (0.7 * numpy.log(theta / arguments.density_soil) + 0.4) * 10 ** (0.6243 * theta)  # Eq. 3.18
        else:
            lam_kersten = 0.1442 * (0.9 * numpy.log(theta / arguments.density_soil) - 0.2) * 10 ** (0.6243 * theta)  # Eq. 3.19
        return lam_kersten

    @staticmethod
    def johansen(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Johansen Model (1977) to compute the thermal conductivity of soil based on
        its volumetric water content and other properties.
        """
        lam_w = thermal_conductivity_water = 0.57
        phi_q = 0.5 * arguments.percentage_sand / 100  # Convert percentage to fraction
        lam_q = 7.7
        lam_other = 2 if phi_q >= 0.2 else 3

        # Calculate porosity
        phi = 1 - arguments.density_soil / arguments.particle_density

        # Thermal conductivity of soil mineral solids
        lam_s = lam_q ** phi_q * lam_other ** (1 - phi_q)

        # Calculate lambda_dry
        lam_dry = (0.135 * arguments.density_soil + 64.7) / (arguments.particle_density - 0.947 * arguments.density_soil)

        # Calculate lambda_sat
        lam_sat = lam_w ** phi * lam_s ** (1 - phi)

        # Saturation degree
        s = theta / phi

        # Considering given information is cut-off, using basic equation for lambda
        lam = lam_dry + s * (lam_sat - lam_dry)  # Using the principle of Kersten number (Ke)

        return lam

    @staticmethod
    def cote_konrad(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Côté & Konrad model (2005) to compute the thermal conductivity of soil based on
        its volumetric water content and other properties.
        """
        k_values = {
            "Gravel and coarse sand":   4.6,
            "Medium and fine sand":     3.55,
            "Silty and clayey soils":   1.9,
            "Organic fibrous soils":    0.6,
        }

        soil_material = "Silty and clayey soils"

        if soil_material not in k_values:
            raise ValueError(f"Invalid soil material: {soil_material}. Supported values are: {', '.join(k_values.keys())}")

        k = k_values[soil_material]
        steps = theta / arguments.porosity_ratio
        ke_cote_konrad = k * steps * (1. + (k - 1.) * steps)

        chi = 0.75
        eta = 1.2
        lam_dry = chi * 10 ** (-eta * arguments.porosity_ratio)

        lam_cote_konrad = lam_dry * ke_cote_konrad
        return lam_cote_konrad

    @staticmethod
    def yang(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        This function implements the Yang et al. (2005) model to compute the thermal conductivity of soil based on its
        volumetric water content and other properties.
        """
        k_t = 0.36
        steps = theta / arguments.porosity_ratio
        ke_yang = numpy.exp(k_t * (1 - 1 / steps))
        phi_quartz = arguments.percentage_sand / 100  # Assuming sand content represents quartz content
        lam_sat = 0.5 ** arguments.porosity_ratio * (7.7 ** phi_quartz * 2 ** (1 - phi_quartz)) ** (1 - arguments.porosity_ratio)
        lam_dry = (.135 * arguments.density_soil + 64.7) / (arguments.particle_density - .947 * arguments.density_soil)  # As used before for dry soil
        lam_yang = lam_dry + ke_yang * (lam_sat - lam_dry)
        return lam_yang

    @classmethod
    def get_static_methods(cls) -> list[Callable[..., any]]:
        return [
            value for name, value in inspect.getmembers(cls)
            if isinstance(inspect.getattr_static(cls, name), staticmethod) and not name.startswith("_")
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
    path = Path("data/")

    # measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-02-21.xlsx"
    # (absolute dichte pro messreihe), volumenanteil wasser pro messung, wärmeleitfähigkeit pro messung
    # measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-04-06.xlsx"
    measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-07-10.xlsx"

    soils_output_file = path / "02_16_Ergebnisse.xlsx"
    soils_output_handler = pandas.ExcelWriter(soils_output_file)

    measurements_output_file = path / "model_fit.xlsx"
    measurements_output_handler = pandas.ExcelWriter(measurements_output_file)

    cmap = pyplot.get_cmap("Set1")

    particle_density = 2650  # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html
    thermal_conductivity_quartz = 7.7  # metall?

    methods = Methods.get_static_methods()

    scatter_data = init_scatter_data(methods)
    measurement_output = init_outputs(methods)

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    for row_index, (n, row) in enumerate(overview_sheet.iterrows()):
        row_index = int(row_index)
        if row_index + 1 == 30:
            pass

        # get cells starting from the 7th column and the 2nd row to the last row
        each_range_str = row[7]
        each_range = tuple(float(x) / 100. for x in each_range_str.split(","))

        each_sheet = data_measurement_sheets.get(f"{row_index + 1:d}")
        if each_sheet is None:
            print(f"Sheet {row_index + 1:d} not found")
            break

        # KA5 name, anteil sand, anteil schluff, anteil lehm, dichte
        short_name, percentage_sand, percentage_silt, percentage_clay, density_soil_non_si, soil_type = row.values[1:7]
        measurement_type = row.values[10]  # Messungstyp
        density_soil = density_soil_non_si * 1000.  # g/cm3 -> kg/m3
        porosity_ratio = 1. - density_soil / particle_density
        print(f"{row_index + 1:d} \t fSand={percentage_sand:.0f}, fSilt={percentage_silt:.0f}, fClay={percentage_clay:.0f}")

        # volumetrischer Sättigungswassergehalt [m3/m3]
        print(porosity_ratio)

        bound_lo = min(each_range)
        bound_hi = max(each_range)

        theta_array = each_sheet["θ [cm3/cm3]"].to_numpy()
        is_punctual = "punctual" in measurement_type.lower()
        is_tu = "t/u" in soil_type.lower()
        is_in_range = (theta_array >= bound_lo) & (bound_hi >= theta_array)

        lambda_array = each_sheet["λ [W/(m∙K)]"].to_numpy()
        filter_array = (
                numpy.isfinite(lambda_array)
                & (0 < theta_array)
            # & numpy.array([not is_punctual] * len(lambda_array))
            # & numpy.array([not is_tu] * len(lambda_array))
            # & is_in_range
        )
        theta_measurement = theta_array[filter_array]
        # theta_measurement = each_sheet["θ [cm3/cm3]"]
        lambda_measurement = lambda_array[filter_array]

        if len(theta_measurement) < 1 or len(lambda_measurement) < 1:
            print(f"Skipping {row_index + 1:d} due to missing data")
            continue

        # Sättigung
        steps = numpy.linspace(1, 0, num=50, endpoint=False)[::-1]

        # Wassergehalt
        theta_range = steps * porosity_ratio
        # theta_range = numpy.linspace(bound_lo, bound_hi, num=50)

        theta_quartz = .5 * percentage_sand / 100.
        thermal_conductivity_other = 3. if theta_quartz < .2 else 2.
        thermal_conductivity_sand = thermal_conductivity_quartz ** theta_quartz * thermal_conductivity_other ** (1 - theta_quartz)  # thermal conductivity of stone? soil?

        no_measurements = len(lambda_measurement)
        measurement_output["Messreihe"].append(row_index + 1)
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

        # START measurements
        for each_method in methods:
            ideal = each_method(theta_measurement, arguments)
            sse = numpy.sum((lambda_measurement - ideal) ** 2)
            measurement_output[f"RMSE {each_method.__name__}"].append(numpy.sqrt(sse / no_measurements))
            scatter_data[each_method.__name__]["model"].extend(ideal)
            scatter_data[each_method.__name__]["data"].extend(lambda_measurement)
            scatter_data[each_method.__name__]["is_punctual"].extend(measurement_type_sequence)
            scatter_data[each_method.__name__]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
            scatter_data[each_method.__name__]["is_tu"].extend(soil_type_sequence)
            average_dir = numpy.sum(ideal - lambda_measurement) / no_measurements
            measurement_output[f"BIAS {each_method.__name__}"].append(average_dir)
        # END measurements

        # START ideal values
        soil_output = {f"Feuchte {row_index + 1:d}, {short_name:s} [m³%]": theta_range}

        # pyplot.figure()
        # pyplot.title(short_name)
        for i, each_method in enumerate(methods):
            lambda_values = each_method(theta_range, arguments)
            # pyplot.plot(theta_range, lambda_values, c=cmap(i), label=each_method.__name__)
            soil_output[f"{each_method.__name__} {row_index + 1:d}, {short_name:s} [W/(mK)]"] = lambda_values

        # pyplot.xlabel("Theta [m³%]")
        # pyplot.ylabel("Lambda [W/(mK)]")
        # pyplot.legend()

        soil_df = pandas.DataFrame(soil_output)
        soil_df.to_excel(soils_output_handler, sheet_name=f"{row_index + 1:d} {short_name:s}")
        # END ideal values

    # START write model fit
    measurements_df = pandas.DataFrame(measurement_output)
    measurements_df.to_excel(measurements_output_handler, index=False)
    measurements_output_handler.close()
    soils_output_handler.close()
    # END write model fit

    # START plot measurements against models
    for method, info in scatter_data.items():
        print(f"{method:s}: scatterplotting...")
        pyplot.figure()
        pyplot.xlabel("Messung [W/(mK)]")
        pyplot.ylabel("Modell [W/(mK)]")
        direction = 0.
        measurements = 0
        for model, data in zip(info["model"], info["data"]):
            delta = model - data
            if not numpy.isnan(delta):
                direction += delta
                measurements += 1

        # todo: is correct?
        pyplot.title(f"{method:s} (bias: {direction / measurements:.2f})")
        pyplot.plot([0, 3], [0, 3], c="black", linestyle="--", alpha=.3)

        non_punctual_x = [each_x for each_x, each_is_punctual in zip(info["data"], info["is_punctual"]) if not each_is_punctual]
        non_punctual_y = [each_y for each_y, each_is_punctual in zip(info["model"], info["is_punctual"]) if not each_is_punctual]
        pyplot.scatter(non_punctual_x, non_punctual_y, c="blue", alpha=.1, s=.5)

        punctual_x = [each_x for each_x, each_is_punctual in zip(info["data"], info["is_punctual"]) if each_is_punctual]
        punctual_y = [each_y for each_y, each_is_punctual in zip(info["model"], info["is_punctual"]) if each_is_punctual]
        pyplot.scatter(punctual_x, punctual_y, c="black", alpha=.8, s=8, linewidths=1, marker="x")

        pyplot.xlim(0, 3)
        pyplot.ylim(0, 3)
        pyplot.savefig(f"plots/scatter_{method:s}.pdf")

    pyplot.show()
    # pyplot.close()
    # END plot measurements against models


if __name__ == "__main__":
    main()
