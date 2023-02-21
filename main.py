# !/usr/bin/env python3
# coding=utf-8
from pathlib import Path

import numpy
from matplotlib import pyplot
import pandas


def hu(S, lam_s, lam_w, porosity, rho_s, rho_si):
    ke_hu = .9878 + .1811 * numpy.log(S)
    # lambda_s = 3.35
    # lambda_w = 0.6
    # lambda_air = 0.0246
    lam_dry = (0.135 * rho_si + 64.7) / (rho_s - 0.947 * rho_si)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_hu = lam_dry + ke_hu * (lam_sat - lam_dry)
    return lam_hu


def markle(S, lam_s, porosity, lam_w, rho_s, rho_si):
    zeta = 8.9
    ke_ma = 1. - numpy.exp(-zeta * S)
    lam_dry = (.135 * rho_si + 64.7) / (rho_s - .947 * rho_si)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_markle = lam_dry + ke_ma * (lam_sat - lam_dry)
    return lam_markle


def brakelmann(S, f_clay, f_sand, f_silt, lam_w, porosity):
    lam_b = .0812 * f_sand + .054 * f_silt + .02 * f_clay
    # rho_p = 0.0263 * f_sand + 0.0265 * f_silt + 0.028 * f_clay
    lam_brakelmann = lam_w ** porosity * lam_b ** (1. - porosity) * numpy.exp(-3.08 * porosity * (1. - S) ** 2)
    return lam_brakelmann


def markert(theta, f_clay, f_sand, porosity, rho):
    p = 1.21, -1.55, .02, .25, 2.29, 2.12, -1.04, -2.03
    lambda_dry = p[0] + p[1] * porosity
    alpha = p[2] * f_clay / 100. + p[3]
    beta = p[4] * f_sand / 100. + p[5] * rho + p[6] * f_sand / 100. * rho + p[7]
    lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))
    return lam_markert


def main() -> None:
    path = Path("data/")

    soils_input_file = path / "23_02_Boeden_Mario.xlsx"                 # absolute dichte pro messreihe
    measurements_input_file = path / "Messdatenbank_FAU.xlsx"           # (absolute dichte pro messreihe), volumenanteil wasser pro messung, wärmeleitfähigkeit pro messung
    output_file = path / "02_16_Ergebnisse.xlsx"

    output_handler = pandas.ExcelWriter(output_file)

    data = pandas.read_excel(soils_input_file, index_col=0)
    cmap = pyplot.get_cmap("Set1")

    density_s = 2650                    # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html
    thermal_conductivity_water = .57    # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
    thermal_conductivity_q = 7.7        # metall?

    for n, col in enumerate(data.columns):
        short_name, percentage_sand, percentage_silt, percentage_clay, density_soil_non_si = data[col].values  # KA5 name, anteil sand, anteil schluff, anteil lehm, dichte
        density_soil = density_soil_non_si * 1000.  # g/cm3 -> kg/m3
        porosity_ratio = 1. - density_soil / density_s
        print(f"{col:d} \t fSand={percentage_sand:d}, fSilt={percentage_silt:d}, fClay={percentage_clay:d}")

        # volumetrischer Sättigungswassergehalt [m3/m3]
        print(porosity_ratio)

        # S_target = theta_measurement / theta_sat

        # Sättigung
        # steps = numpy.linspace(1e-6, 1, num=50)
        steps = numpy.linspace(1, 0, num=50, endpoint=False)[::-1]

        # Wassergehalt
        theta_range = steps * porosity_ratio

        phi_q = .5 * percentage_sand / 100.
        thermal_conductivity_other = 3. if phi_q < .2 else 2.
        thermal_conductivity_s = thermal_conductivity_q ** phi_q * thermal_conductivity_other ** (1 - phi_q)  # thermal conductivity of stone? soil?

        # Modelle
        lambda_markert = markert(
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
            thermal_conductivity_s,
            porosity_ratio,
            thermal_conductivity_water,
            density_s,
            density_soil
        )

        lambda_hu = hu(
            steps,
            thermal_conductivity_s,
            thermal_conductivity_water,
            porosity_ratio,
            density_s,
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
        pyplot.ylabel("Lambda [W/mK]")
        pyplot.legend()

        # write to file
        output = {
            f"Feuchte {n + 1:d}, {short_name:s} [m³%]": theta_range,
            f"Markert {n + 1:d}, {short_name:s} [W/mK]": lambda_markert,
            f"Brakelmann {n + 1:d}, {short_name:s} [W/mK]": lambda_brakelmann,
            f"Markle {n + 1:d}, {short_name:s} [W/mK]": lambda_markle,
            f"Hu {n + 1:d}, {short_name:s} [W/mK]": lambda_hu
        }

        df = pandas.DataFrame(output)
        df.to_excel(output_handler, sheet_name=f"{n + 1:d} {short_name:s}")

    # write xls
    output_handler.close()
    pyplot.show()


if __name__ == "__main__":
    main()
