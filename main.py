# !/usr/bin/env python3
# coding=utf-8

import numpy
import os
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


def markle(S, f_sand, porosity, rho_s, rho_si):
    zeta = 8.9
    ke_ma = 1. - numpy.exp(-zeta * S)
    lam_dry = (.135 * rho_si + 64.7) / (rho_s - .947 * rho_si)
    lam_w = .57
    phi_q = .5 * f_sand / 100.
    lam_q = 7.7
    if phi_q < .2:
        lam_other = 3
    else:
        lam_other = 2
    lam_s = lam_q ** phi_q * lam_other ** (1 - phi_q)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_markle = lam_dry + ke_ma * (lam_sat - lam_dry)
    return lam_markle, lam_s, lam_w


def brakelmann(S, f_clay, f_sand, f_silt, lam_w, porosity):
    lam_b = .0812 * f_sand + .054 * f_silt + .02 * f_clay
    # rho_p = 0.0263 * f_sand + 0.0265 * f_silt + 0.028 * f_clay
    lam_brakelmann = lam_w ** porosity * lam_b ** (1. - porosity) * numpy.exp(-3.08 * porosity * (1. - S) ** 2)
    return lam_brakelmann


def markert(f_clay, f_sand, porosity, rho, theta):
    p = 1.21, -1.55, .02, .25, 2.29, 2.12, -1.04, -2.03
    lambda_dry = p[0] + p[1] * porosity
    alpha = p[2] * f_clay / 100. + p[3]
    beta = p[4] * f_sand / 100. + p[5] * rho + p[6] * f_sand / 100. * rho + p[7]
    lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))
    return lam_markert


def main() -> None:
    path = "data/"

    input_file = os.path.join(path, "23_02_Boeden_Mario.xlsx")
    output_file = pandas.ExcelWriter(os.path.join(path, "02_16_Ergebnisse.xlsx"))

    data = pandas.read_excel(input_file, index_col=0)
    cmap = pyplot.get_cmap("Set1")

    rho_s = 2650
    lam_w = .57

    for n, col in enumerate(data.columns):
        short_name, f_sand, f_silt, f_clay, rho = data[col].values
        rho_si = rho * 1000.  # g/cm3 -> kg/m3
        porosity = 1. - rho_si / rho_s
        print(f"{col:d} \t fSand={f_sand:d}, fSilt={f_silt:d}, fClay={f_clay:d}")

        # volumetrischer Sättigungswassergehalt [m3/m3]
        theta_sat = porosity
        print(theta_sat)

        # S_target = theta_measurement / theta_sat

        # Sättigung
        S = numpy.linspace(1e-6, 1, num=50)

        # Wassergehalt
        theta = S * theta_sat

        # Markert
        lam_markert = markert(f_clay, f_sand, porosity, rho, theta)

        # Brakelmann
        lam_brakelmann = brakelmann(S, f_clay, f_sand, f_silt, lam_w, porosity)

        # Markle model
        lam_markle, lam_s, lam_w = markle(S, f_sand, porosity, rho_s, rho_si)

        # Hu model
        lam_hu = hu(S, lam_s, lam_w, porosity, rho_s, rho_si)

        # plot
        pyplot.figure()
        pyplot.title(short_name)
        pyplot.plot(theta, lam_markert, c=cmap(0), label='Markert')
        pyplot.plot(theta, lam_markle, c=cmap(1), label="Markle")
        pyplot.plot(theta, lam_brakelmann, c=cmap(2), label='Brakelmann')
        pyplot.plot(theta, lam_hu, c=cmap(3), label='Hu')
        pyplot.xlabel('Theta [m3/m3]')
        pyplot.ylabel('Lambda [W/mK]')
        pyplot.legend()

        # write to file
        output = dict()
        output[f'Feuchte {n + 1:d}, {short_name:s} [m³/m³]'] = theta
        output[f'Markert {n + 1:d}, {short_name:s} [W/mK]'] = lam_markert
        output[f'Brakelmann {n + 1:d}, {short_name:s} [W/mK]'] = lam_brakelmann
        output[f'Markle {n + 1:d}, {short_name:s} [W/mK]'] = lam_markle
        output[f'Hu {n + 1:d}, {short_name:s} [W/mK]'] = lam_hu

        df = pandas.DataFrame(output)
        df.to_excel(output_file, sheet_name=f"{n + 1:d} {short_name:s}")

    # # write xls
    output_file.close()
    pyplot.show()


if __name__ == "__main__":
    # required:
    #   - messungen, jeweils: absolute wassersättigung und wärmeleitfähigkeit
    main()
