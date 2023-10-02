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
    def hu(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
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
        beta = p[4] * arguments.percentage_sand / 100. + p[5] * arguments.density_soil_non_si + p[6] * arguments.percentage_sand / 100. * arguments.density_soil_non_si + p[7]
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
    def _untested_devries(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        - Some assumptions are made here, especially about how \( \Theta_k \) is used.
        - The data for \( \Lambda_i \) and \( g_i \) are directly taken from the table provided.
        - Adjustments might be needed based on more specifics about the DeVries model and the usage of the equations, especially about the range conditions for \( \Theta_L \).
        - The model uses only the data and equations provided, and further details (like how \( \Theta_k \) is determined) would be needed to complete it.
        - Ensure that the provided data is sufficient and accurately represented in the code.
        """

        # Constants from the table
        lambda_water = 0.57
        lambda_vapor_air = 0.035 + 0.298 * theta / arguments.porosity_ratio
        lambda_quartz = 8.8
        lambda_other_minerals = 2.0
        lambda_organic_matter = 0.25

        # Shape factors from the table
        g = [0, 0, 0.125, 0.125, 0.5]

        # Computing k values using equation (4)
        k = [1]
        for i in range(1, 5):
            lambda_i = [lambda_water, lambda_vapor_air, lambda_quartz, lambda_other_minerals, lambda_organic_matter][i]
            g_i = g[i]
            k_val = (2/3) * (1 / (1 + g_i * (lambda_i / lambda_water - 1))) + (1/3) * (1 / (1 + (lambda_i / lambda_water - 1) * (1 - 2 * g_i)))
            k.append(k_val)

        # I REMOVED IT. Computing lambda_dry using equation (3)
        # weird_sum = sum([k[i] * [Lambda_water, Lambda_vapor_air, Lambda_quartz, Lambda_other_minerals, Lambda_organic_matter][i] for i in range(5)])
        # weird_factor = (arguments.porosity_ratio * Lambda_vapor_air + weird_sum)
        # lambda_dry = 1.25 * weird_factor / (arguments.porosity_ratio + sum([k[i] for i in range(5)]))

        # Using the equation (1) to compute lambda, ensuring operations are vectorized
        numerator = sum([k[i] * [lambda_water, lambda_vapor_air, lambda_quartz, lambda_other_minerals, lambda_organic_matter][i] for i in range(5)]) * theta
        denominator = sum([k[i] * theta for i in range(5)])
        lambda_result = numerator / denominator

        return lambda_result

    @staticmethod
    def _untested_sadeghi(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        I made assumptions about certain variables (like `lam_sat`, `theta_c`, and `theta_s`) as they were not clearly defined in the provided context. You might
        want to adjust these based on more detailed information or context.
        """

        lam_sat = arguments.thermal_conductivity_sand  # Assuming that the thermal conductivity of sand is equivalent to lam_sat as it's not clearly defined
        lam_dry = (.135 * arguments.density_soil + 64.7) / (arguments.particle_density - .947 * arguments.density_soil)

        theta_c = arguments.porosity_ratio  # Assuming that porosity_ratio is equivalent to theta_c as it's not clearly defined
        theta_s = 1.0  # Assuming full saturation value for theta_s as it's not clearly defined in the given context

        t_s = (lam_sat ** (1 / theta_s))
        t = 1 / t_s

        a_1 = (theta_c * lam_sat ** t - (theta_s - theta_c) * lam_dry ** t) / (lam_sat ** t - lam_dry ** t)
        a_2 = (theta_s - theta_c) / (lam_sat ** t - lam_dry ** t)
        a_3 = -theta_c * lam_sat ** t * lam_dry ** t / (lam_sat ** t - lam_dry ** t)

        lambda_values = ((- (a_1 - theta) + numpy.sqrt((a_1 - theta) ** 2 - 4 * a_2 * a_3)) / (2 * a_2)) ** t_s

        return lambda_values

    @staticmethod
    def _untested_lu_et_al(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        seems fine
        """
        # Extract the parameters from arguments
        f_sand = arguments.percentage_sand
        f_clay = arguments.percentage_clay
        rho_b = arguments.density_soil_non_si

        # Implement the provided equations
        lambda_dry = 0.51 - 0.56 * arguments.porosity_ratio
        delta = 0.67 * f_clay + 0.24
        beta = 1.97 * f_sand + 1.87 * rho_b - 1.36 * rho_b * f_sand - 0.95

        lambda_theta = numpy.exp(beta - theta ** (-delta)) + lambda_dry
        return lambda_theta

    @staticmethod
    def _untested_kersten_model(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        seems fine
        """
        # Determine if the soil contains more than 50% sand or less
        is_sandy = (arguments.percentage_sand / 100) > 0.5

        # Apply the respective formula based on sandiness
        if is_sandy:
            lambda_val = 0.1442 * (0.7 * numpy.log(theta / arguments.density_soil) + 0.4) * 10 ** (0.6243 * theta)
        else:
            lambda_val = 0.1442 * (0.9 * numpy.log(theta / arguments.density_soil) - 0.2) * 10 ** (0.6243 * theta)

        return lambda_val

    @staticmethod
    def _untested_johansen(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        seems fine
        """
        # Constants
        lam_w = 0.57  # Thermal conductivity of water

        # Porosity
        phi = 1 - arguments.density_soil / arguments.particle_density

        # Thermal conductivity of natural dry soils
        lam_dry = (0.135 * arguments.density_soil + 64.7) / (arguments.particle_density - 0.947 * arguments.density_soil)

        # Volume fraction of quartz
        phi_q = 0.5 * arguments.percentage_sand

        # Thermal conductivity of solids
        if phi_q < 0.2:
            lam_other = 3.0  # W·m{-1}°·K{-1}
        else:
            lam_other = 2.0  # W·m{-1}°·K{-1}
        lam_q = 7.7  # W·m{-1}°·K{-1} for quartz
        lam_s = lam_q ** phi_q * lam_other ** (1 - phi_q)

        # Thermal conductivity of unfrozen soils at saturation
        lam_sat = lam_w ** phi * lam_s ** (1 - phi)

        # Saturations
        S = theta / arguments.porosity_ratio

        # Formalism for interpolation between lam_dry and lam_sat
        Ke = numpy.where(S > 0.05, 1 + 0.7 * numpy.log10(S), 1 + numpy.log10(S))

        # Thermal conductivity of the soil
        lam = lam_dry + Ke * (lam_sat - lam_dry)

        return lam

    @staticmethod
    def _untested_cote_konrad(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        I've inferred the classification of soil types based on the given percentages for sand, silt, and clay. You may want to adjust the thresholds based on more
        accurate soil classifications if available.
        """
        # Define constants
        chi = 0.75
        eta = 1.2

        # Porosity
        phi = 1 - arguments.density_soil / arguments.particle_density

        # Derive coefficient k based on soil type
        if arguments.percentage_sand >= 50:  # Assuming Gravel and coarse sand have high sand percentage
            k = 4.6
        elif 10 <= arguments.percentage_sand < 50:  # Assuming Medium and fine sand criteria
            k = 3.55
        elif arguments.percentage_clay > 50 or arguments.percentage_silt > 50:  # Silty and clayey soils
            k = 1.9
        else:  # Organic fibrous soils or any other types not explicitly stated
            k = 0.6

        # Compute the Kersten coefficient Ke
        S = theta / arguments.porosity_ratio
        Ke = k * S * (1 + (k - 1) * S)

        # Estimate thermal conductivity at the dry end
        lam_dry = chi * 10 ** (-eta * phi)

        # For the given model, the thermal conductivity estimation is based on the Johansen model with a modified Kersten coefficient
        # Thermal conductivity of the soil using the Johansen equation
        lam = lam_dry + Ke * (arguments.thermal_conductivity_sand - lam_dry)

        return lam

    @staticmethod
    def yang(theta: numpy.ndarray, arguments: Arguments) -> numpy.ndarray:
        """
        - The content of quartz ϕ quartz is inferred to be directly proportional to the sand percentage. Adjust this if there's a more accurate relation or data
        available.
        - I took λ dry from the previous Markle model. Adjust this if a different dry thermal conductivity is desired or if λ dry is provided explicitly for
        this model.
        """
        # Constants
        k_T = 0.36
        S = theta / arguments.porosity_ratio
        phi = 1 - arguments.density_soil / arguments.particle_density
        phi_quartz = arguments.percentage_sand / 100  # Assuming quartz content is directly proportional to sand percentage

        # Interpolation coefficient Ke
        Ke = numpy.exp(k_T * (1 - 1 / S))

        # Thermal conductivity at the dry end, inferred from other models
        lam_dry = (.135 * arguments.density_soil + 64.7) / (arguments.particle_density - .947 * arguments.density_soil)

        # Estimate thermal conductivity at saturation
        lam_sat = (0.5 ** phi) * (7.7 ** phi_quartz * 2 ** (1 - phi_quartz)) ** (1 - phi)

        # Compute thermal conductivity
        lam_yang = lam_dry + Ke * (lam_sat - lam_dry)

        return lam_yang

    @classmethod
    def get_static_methods(cls) -> list[Callable[..., any]]:
        return [
            value for name, value in inspect.getmembers(cls)
            if isinstance(inspect.getattr_static(cls, name), staticmethod) and not name.startswith("_")
        ]


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

    particle_density = 2650             # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html
    thermal_conductivity_quartz = 7.7   # metall?

    methods = Methods.get_static_methods()

    scatter_data = {
        each_method.__name__:
            {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []}
        for each_method in methods
    }

    measurement_output = {
        "Messreihe":             [],
        "#Messungen":            []
    }
    for each_method in methods:
        measurement_output[f"RMSE {each_method.__name__}"] = []
        measurement_output[f"DIR {each_method.__name__}"] = []

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    for row_index, (n, row) in enumerate(overview_sheet.iterrows()):
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
            markert_avrg_dir = numpy.sum(ideal - lambda_measurement) / no_measurements
            measurement_output[f"DIR {each_method.__name__}"].append(markert_avrg_dir)
        # END measurements

        # START ideal values
        soil_output = {f"Feuchte {row_index + 1:d}, {short_name:s} [m³%]": theta_range}

        pyplot.figure()
        pyplot.title(short_name)
        for i, each_method in enumerate(methods):
            lambda_values = each_method(theta_range, arguments)
            pyplot.plot(theta_range, lambda_values, c=cmap(i), label=each_method.__name__)
            soil_output[f"{each_method.__name__} {row_index + 1:d}, {short_name:s} [W/(mK)]"] = lambda_values

        pyplot.xlabel("Theta [m³%]")
        pyplot.ylabel("Lambda [W/(mK)]")
        pyplot.legend()

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

        pyplot.title(f"{method:s} (direction: {direction / measurements:.2f})")
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
