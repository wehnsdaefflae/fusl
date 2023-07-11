import numpy

from utils import Model, Available


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


def markert_all(theta, f_clay, f_sand, porosity, rho, p: tuple[float, ...] | None = None):
    if p is None:
        sand_silt_loam = +1.21, -1.55, +0.02, +0.25, +2.29, +2.12, -1.04, -2.03
        p = sand_silt_loam

    lambda_dry = p[0] + p[1] * porosity
    alpha = p[2] * f_clay / 100. + p[3]
    beta = p[4] * f_sand / 100. + p[5] * rho + p[6] * f_sand / 100. * rho + p[7]
    lam_markert = lambda_dry + numpy.exp(beta - theta ** (-alpha))
    return lam_markert


def markert_specific(theta, f_clay, f_silt, f_sand, porosity, rho, packed: bool = True):
    # p table: 1323, texture groups: 1320, https://www.dropbox.com/s/y6hm5m6necbzkpr/Soil%20Science%20Soc%20of%20Amer%20J%20-%202017%20-%20Markert%20-.pdf?dl=0

    tg_sand_u = +1.51, -3.07, +1.24, +0.24, +1.87, +2.34, -1.34, -1.32
    tg_sand_p = +0.72, -0.74, -0.82, +0.22, +1.55, +2.22, -1.36, -0.95

    tg_silt_u = +0.92, -1.08, +0.90, +0.21, +0.14, +1.27, +0.25, -0.33
    tg_silt_p = +1.83, -2.75, +0.12, +0.22, +5.00, +1.32, -1.56, -0.88

    tg_loam_u = +1.24, -1.55, +0.08, +0.28, +4.26, +1.17, -1.62, -1.19
    tg_loam_p = +1.79, -2.62, -0.39, +0.25, +3.83, +1.44, -1.11, -2.02

    if f_silt + 2 * f_clay < 30:
        # TG Sand; S, LS
        p = tg_sand_p if packed else tg_sand_u

    elif 50 < f_silt and f_clay < 27:
        # TG Silt; Si, SiL
        p = tg_silt_p if packed else tg_silt_u

    else:
        # TG Loam; SL, SCL, SiCL, CL, L
        p = tg_loam_p if packed else tg_loam_u

    return markert_all(theta, f_clay, f_sand, porosity, rho, p=p)


def markert_lu(theta, f_clay, f_sand, porosity, rho):
    # seite 19, https://www.dropbox.com/s/6iq8z26iahk6s6d/2018-10-25_FAU_TB_Endbericht_Pos.3.1_V3.pdf?dl=0
    lambda_dry = -.56 * porosity + .51
    sigma = .67 * f_clay / 100. + .24
    beta = 1.97 * f_sand / 100. + rho * 1.87 - 1.36 * f_sand / 100. - .95
    lam_markert_lu = lambda_dry + numpy.exp(beta - theta ** (-sigma))
    return lam_markert_lu

