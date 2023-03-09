# !/usr/bin/env python3
# coding=utf-8
from pathlib import Path

import numpy
from matplotlib import pyplot
import pandas


def hu(steps, lam_s, lam_w, porosity, rho_s, rho_si):
    ke_hu = .9878 + .1811 * numpy.log(steps)
    # lambda_s = 3.35
    # lambda_w = 0.6
    # lambda_air = 0.0246
    lam_dry = (0.135 * rho_si + 64.7) / (rho_s - 0.947 * rho_si)
    ke_hu[ke_hu == numpy.inf] = lam_dry
    ke_hu[ke_hu == -numpy.inf] = lam_dry
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_hu = lam_dry + ke_hu * (lam_sat - lam_dry)
    return lam_hu


def markle(steps, lam_s, porosity, lam_w, rho_s, rho_si):
    zeta = 8.9
    ke_ma = 1. - numpy.exp(-zeta * steps)
    lam_dry = (.135 * rho_si + 64.7) / (rho_s - .947 * rho_si)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_markle = lam_dry + ke_ma * (lam_sat - lam_dry)
    return lam_markle


def brakelmann(steps, f_clay, f_sand, f_silt, lam_w, porosity):
    lam_b = .0812 * f_sand + .054 * f_silt + .02 * f_clay
    # rho_p = 0.0263 * f_sand + 0.0265 * f_silt + 0.028 * f_clay
    lam_brakelmann = lam_w ** porosity * lam_b ** (1. - porosity) * numpy.exp(-3.08 * porosity * (1. - steps) ** 2)
    return lam_brakelmann


def markert_all(theta, f_clay, f_sand, porosity, rho):
    p = 1.21, -1.55, .02, .25, 2.29, 2.12, -1.04, -2.03
    lambda_dry = p[0] + p[1] * porosity
    alpha = p[2] * f_clay / 100. + p[3]
    beta = p[4] * f_sand / 100. + p[5] * rho + p[6] * f_sand / 100. * rho + p[7]
    lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))
    return lam_markert


def markert_specific():
    # https://www.dropbox.com/s/y6hm5m6necbzkpr/Soil%20Science%20Soc%20of%20Amer%20J%20-%202017%20-%20Markert%20-.pdf?dl=0
    pass


def markert_lu():
    # https://www.dropbox.com/s/6iq8z26iahk6s6d/2018-10-25_FAU_TB_Endbericht_Pos.3.1_V3.pdf?dl=0
    pass


def main() -> None:
    path = Path("data/")

    soils_input_file = path / "23_02_Boeden_Mario.xlsx"                 # absolute dichte pro messreihe
    soils_output_file = path / "02_16_Ergebnisse.xlsx"
    soils_output_handler = pandas.ExcelWriter(soils_output_file)
    data_soil = pandas.read_excel(soils_input_file, index_col=0)

    measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-02-21.xlsx"   # (absolute dichte pro messreihe), volumenanteil wasser pro messung, wärmeleitfähigkeit pro messung
    measurements_output_file = path / "model_fit.xlsx"
    measurements_output_handler = pandas.ExcelWriter(measurements_output_file)
    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)

    cmap = pyplot.get_cmap("Set1")

    particle_density = 2650             # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html
    thermal_conductivity_water = .57    # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
    thermal_conductivity_quartz = 7.7   # metall?

    measurement_output = {
        "Messreihe":                                   [],
        "#Messungen":                                  [],
        "Normierter quadratischer Fehler Markert":     [],
        "Normierter quadratischer Fehler Brakelmann":  [],
        "Normierter quadratischer Fehler Markle":      [],
        "Normierter quadratischer Fehler Hu":          []

    }

    for n, col in enumerate(data_soil.columns):
        each_sheet = data_measurement_sheets.get(f"{n+1:d}")
        if each_sheet is None:
            print(f"Sheet {n+1:d} not found")
            break

        short_name, percentage_sand, percentage_silt, percentage_clay, density_soil_non_si = data_soil[col].values  # KA5 name, anteil sand, anteil schluff, anteil lehm, dichte
        density_soil = density_soil_non_si * 1000.  # g/cm3 -> kg/m3
        porosity_ratio = 1. - density_soil / particle_density
        print(f"{col:d} \t fSand={percentage_sand:.0f}, fSilt={percentage_silt:.0f}, fClay={percentage_clay:.0f}")

        # volumetrischer Sättigungswassergehalt [m3/m3]
        print(porosity_ratio)

        theta_measurement = each_sheet["θ [cm3/cm3]"]
        lambda_measurement = each_sheet["λ [W/(m∙K)]"]
        step_measurement = theta_measurement / porosity_ratio

        # Sättigung
        steps = numpy.linspace(1, 0, num=50, endpoint=False)[::-1]

        # Wassergehalt
        theta_range = steps * porosity_ratio

        theta_quartz = .5 * percentage_sand / 100.
        thermal_conductivity_other = 3. if theta_quartz < .2 else 2.
        thermal_conductivity_sand = thermal_conductivity_quartz ** theta_quartz * thermal_conductivity_other ** (1 - theta_quartz)  # thermal conductivity of stone? soil?

        # Modellpassung
        lambda_markert_ideal = markert_all(
            theta_measurement,
            percentage_clay,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si
        )

        lambda_brakelmann_ideal = brakelmann(
            step_measurement,
            percentage_clay,
            percentage_sand,
            percentage_silt,
            thermal_conductivity_water,
            porosity_ratio)

        lambda_markle_ideal = markle(
            step_measurement,
            thermal_conductivity_sand,
            porosity_ratio,
            thermal_conductivity_water,
            particle_density,
            density_soil
        )

        lambda_hu_ideal = hu(
            step_measurement,
            thermal_conductivity_sand,
            thermal_conductivity_water,
            porosity_ratio,
            particle_density,
            density_soil
        )

        markert_quadratic_error = numpy.sum((lambda_measurement - lambda_markert_ideal) ** 2)
        brakelmann_quadratic_error = numpy.sum((lambda_measurement - lambda_brakelmann_ideal) ** 2)
        markle_quadratic_error = numpy.sum((lambda_measurement - lambda_markle_ideal) ** 2)
        hu_quadratic_error = numpy.sum((lambda_measurement - lambda_hu_ideal) ** 2)

        # hu nicht definiert für wasseranteil <= .0?

        no_measurements = len(lambda_measurement)
        measurement_output["Messreihe"].append(n + 1)
        measurement_output["#Messungen"].append(no_measurements)
        measurement_output["Normierter quadratischer Fehler Markert"].append(markert_quadratic_error / no_measurements)
        measurement_output["Normierter quadratischer Fehler Brakelmann"].append(brakelmann_quadratic_error / no_measurements)
        measurement_output["Normierter quadratischer Fehler Markle"].append(markle_quadratic_error / no_measurements)
        measurement_output["Normierter quadratischer Fehler Hu"].append(hu_quadratic_error / no_measurements)

        # Modelle
        lambda_markert = markert_all(
            theta_range,
            percentage_clay,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si
        )

        lambda_brakelmann = brakelmann(
            steps,
            percentage_clay,
            percentage_sand,
            percentage_silt,
            thermal_conductivity_water,
            porosity_ratio)

        lambda_markle = markle(
            steps,
            thermal_conductivity_sand,
            porosity_ratio,
            thermal_conductivity_water,
            particle_density,
            density_soil
        )

        lambda_hu = hu(
            steps,
            thermal_conductivity_sand,
            thermal_conductivity_water,
            porosity_ratio,
            particle_density,
            density_soil
        )

        # plot
        pyplot.figure()
        pyplot.title(short_name)
        pyplot.plot(theta_range, lambda_markert, c=cmap(0), label="Markert")
        pyplot.plot(theta_range, lambda_markle, c=cmap(1), label="Markle")
        pyplot.plot(theta_range, lambda_brakelmann, c=cmap(2), label="Brakelmann")
        pyplot.plot(theta_range, lambda_hu, c=cmap(3), label="Hu")
        pyplot.xlabel("Theta [m³%]")
        pyplot.ylabel("Lambda [W/(mK)]")
        pyplot.legend()

        # write to file
        soil_output = {
            f"Feuchte {n + 1:d}, {short_name:s} [m³%]":     theta_range,
            f"Markert {n + 1:d}, {short_name:s} [W/(mK)]":    lambda_markert,
            f"Brakelmann {n + 1:d}, {short_name:s} [W/(mK)]": lambda_brakelmann,
            f"Markle {n + 1:d}, {short_name:s} [W/(mK)]":     lambda_markle,
            f"Hu {n + 1:d}, {short_name:s} [W/(mK)]":         lambda_hu
        }

        soil_df = pandas.DataFrame(soil_output)
        soil_df.to_excel(soils_output_handler, sheet_name=f"{n + 1:d} {short_name:s}")

    # write xls
    measurements_df = pandas.DataFrame(measurement_output)
    measurements_df.to_excel(measurements_output_handler, index=False)
    measurements_output_handler.close()

    soils_output_handler.close()

    pyplot.show()


if __name__ == "__main__":
    main()
