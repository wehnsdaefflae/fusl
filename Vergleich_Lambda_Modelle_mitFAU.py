# coding=utf-8
import numpy
import os
from matplotlib import pyplot
import pandas

path = "data/"

xlsfile = os.path.join(path, "23_02_Boeden_Mario.xlsx")
writer_xls = pandas.ExcelWriter(os.path.join(path, "02_16_Ergebnisse.xlsx"))

data = pandas.read_excel(xlsfile, index_col=0)
cmap = pyplot.get_cmap("Set1")

rho_s = 2650
lam_w = 0.57

for n, col in enumerate(data.columns):

    short_name, f_sand, f_silt, f_clay, rho = data[col].values
    rho_si = rho*1000   # g/cm3 -> kg/m3
    porosity = 1 - rho_si / rho_s
    print("{} \t fSand={}, fSilt={}, fClay={}".format(col, f_sand, f_silt, f_clay))

    # volumetrischer Sättigungswassergehalt [m3/m3]
    theta_sat = porosity
    print(theta_sat)

    # S_target = theta_measurement / theta_sat

    # Sättigung
    S = numpy.linspace(1e-6, 1, num=50)
    # Wassergehalt
    theta = S * theta_sat

    # Markert
    p = [1.21, -1.55, 0.02, 0.25, 2.29, 2.12, -1.04, -2.03]
    lambda_dry = p[0] + p[1] * porosity
    alpha = p[2] * f_clay/100 + p[3]
    beta = p[4] * f_sand/100 + p[5] * rho + p[6] * f_sand/100 * rho + p[7]
    lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))

    # Brakelmann
    lam_b = 0.0812*f_sand + 0.054*f_silt + 0.02*f_clay
    rho_p = 0.0263*f_sand + 0.0265*f_silt + 0.028*f_clay
    lam_brakelmann = lam_w ** porosity * lam_b ** (1 - porosity) * numpy.exp(-3.08 * porosity * (1 - S) ** 2)

    # Markle model
    zeta = 8.9
    Ke_Ma = 1 - numpy.exp(-zeta * S)
    lam_dry = (0.135*rho_si + 64.7) / (rho_s - 0.947*rho_si)
    lam_w = 0.57
    phi_q = 0.5 * f_sand/100
    lam_q = 7.7
    if phi_q<0.2:
        lam_other = 3
    else:
        lam_other = 2
    lam_s = lam_q**phi_q * lam_other**(1-phi_q)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_markle = lam_dry + Ke_Ma * (lam_sat - lam_dry)

    # Hu model
    Ke_Hu = 0.9878 + 0.1811 * numpy.log(S)
    lambda_s = 3.35
    lambda_w = 0.6
    lambda_air = 0.0246
    lam_dry = (0.135 * rho_si + 64.7) / (rho_s - 0.947 * rho_si)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_Hu = lam_dry + Ke_Hu * (lam_sat - lam_dry)

    output = {}
    output[f'Feuchte {n+1}, {short_name} [m³/m³]'] = theta
    output[f'Markert {n+1}, {short_name} [W/mK]'] = lam_markert
    output[f'Brakelmann {n+1}, {short_name} [W/mK]'] = lam_brakelmann
    output[f'Markle {n+1}, {short_name} [W/mK]'] = lam_markle
    output[f'Hu {n+1}, {short_name} [W/mK]'] = lam_Hu

    pyplot.figure()
    pyplot.title(short_name)
    pyplot.plot(theta, lam_markert, c=cmap(0), label='Markert')
    pyplot.plot(theta, lam_markle, c=cmap(1), label="Markle")
    pyplot.plot(theta, lam_brakelmann, c=cmap(2), label='Brakelmann')
    pyplot.plot(theta, lam_Hu, c=cmap(3), label='Hu')
    pyplot.xlabel('Theta [m3/m3]')
    pyplot.ylabel('Lambda [W/mK]')
    pyplot.legend()

    df = pandas.DataFrame(output)
    df.to_excel(writer_xls, sheet_name=f'{n+1} {short_name}')

    # if n>4:
    #     break

# # write xls
writer_xls.save()


pyplot.show()
