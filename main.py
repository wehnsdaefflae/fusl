# !/usr/bin/env python3
# coding=utf-8
from pathlib import Path

import numpy
from matplotlib import pyplot
import pandas

numpy.seterr(all='raise')


def hu(theta_range, lam_s, lam_w, porosity, rho_s, rho_si):
    steps = theta_range / porosity
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


def markle(theta_range, lam_s, porosity, lam_w, rho_s, rho_si):
    zeta = 8.9
    steps = theta_range / porosity
    ke_ma = 1. - numpy.exp(-zeta * steps)
    lam_dry = (.135 * rho_si + 64.7) / (rho_s - .947 * rho_si)
    lam_sat = lam_w ** porosity * lam_s ** (1 - porosity)
    lam_markle = lam_dry + ke_ma * (lam_sat - lam_dry)
    return lam_markle


def brakelmann(theta_range, f_clay, f_sand, f_silt, lam_w, porosity):
    lam_b = .0812 * f_sand + .054 * f_silt + .02 * f_clay
    # rho_p = 0.0263 * f_sand + 0.0265 * f_silt + 0.028 * f_clay
    steps = theta_range / porosity
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


def main() -> None:
    path = Path("data/")

    soils_output_file = path / "02_16_Ergebnisse.xlsx"
    soils_output_handler = pandas.ExcelWriter(soils_output_file)

    # measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-02-21.xlsx"
    # (absolute dichte pro messreihe), volumenanteil wasser pro messung, wärmeleitfähigkeit pro messung
    # measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-04-06.xlsx"
    measurements_input_file = path / "Messdatenbank_FAU_Stand_2023-07-10.xlsx"
    measurements_output_file = path / "model_fit.xlsx"
    measurements_output_handler = pandas.ExcelWriter(measurements_output_file)
    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)

    cmap = pyplot.get_cmap("Set1")

    particle_density = 2650             # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html
    thermal_conductivity_water = .57    # wiki: 0.597 https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers
    thermal_conductivity_quartz = 7.7   # metall?

    scatter_data = {
        "Markert":          {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
        "Brakelmann":       {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
        "Markle":           {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
        "Hu":               {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
        "Markert spez. u.": {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
        "Markert spez. p.": {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
        "Markert Lu":       {"model": [], "data": [], "is_punctual": [], "is_in_range": [], "is_tu": []},
    }

    measurement_output = {
        "Messreihe":             [],
        "#Messungen":            [],
        "RMSE Markert":          [],
        "RMSE Brakelmann":       [],
        "RMSE Markle":           [],
        "RMSE Hu":               [],
        "RMSE Markert spez. u.": [],
        "RMSE Markert spez. p.": [],
        "RMSE Markert Lu":       [],
        "DIR Markert":           [],
        "DIR Brakelmann":        [],
        "DIR Markle":            [],
        "DIR Hu":                [],
        "DIR Markert spez. u.":  [],
        "DIR Markert spez. p.":  [],
        "DIR Markert Lu":        [],
    }

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

        short_name, percentage_sand, percentage_silt, percentage_clay, density_soil_non_si, soil_type = row.values[1:7]  # KA5 name, anteil sand, anteil schluff, anteil lehm, dichte
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
                & numpy.array([not is_punctual] * len(lambda_array))
                & numpy.array([not is_tu] * len(lambda_array))
                & is_in_range
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

        # Modellpassung
        lambda_markert_ideal = markert_all(
            theta_measurement,
            percentage_clay,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si)

        lambda_brakelmann_ideal = brakelmann(
            theta_measurement,
            percentage_clay,
            percentage_sand,
            percentage_silt,
            thermal_conductivity_water,
            porosity_ratio)

        lambda_markle_ideal = markle(
            theta_measurement,
            thermal_conductivity_sand,
            porosity_ratio,
            thermal_conductivity_water,
            particle_density,
            density_soil)

        lambda_hu_ideal = hu(
            theta_measurement,
            thermal_conductivity_sand,
            thermal_conductivity_water,
            porosity_ratio,
            particle_density,
            density_soil)

        lambda_markert_specific_unpacked = markert_specific(
            theta_measurement,
            percentage_clay,
            percentage_silt,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si,
            packed=False)

        lambda_markert_specific_packed = markert_specific(
            theta_measurement,
            percentage_clay,
            percentage_silt,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si,
            packed=True)

        lambda_markert_lu = markert_lu(
            theta_measurement,
            percentage_clay,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si)

        no_measurements = len(lambda_measurement)
        measurement_output["Messreihe"].append(row_index + 1)
        measurement_output["#Messungen"].append(no_measurements)

        markert_sse = numpy.sum((lambda_measurement - lambda_markert_ideal) ** 2)
        brakelmann_sse = numpy.sum((lambda_measurement - lambda_brakelmann_ideal) ** 2)
        markle_sse = numpy.sum((lambda_measurement - lambda_markle_ideal) ** 2)
        hu_sse = numpy.sum((lambda_measurement - lambda_hu_ideal) ** 2)
        markert_specific_unpacked_sse = numpy.sum((lambda_measurement - lambda_markert_specific_unpacked) ** 2)
        markert_specific_packed_sse = numpy.sum((lambda_measurement - lambda_markert_specific_packed) ** 2)
        markert_lu_sse = numpy.sum((lambda_measurement - lambda_markert_lu) ** 2)

        measurement_output["RMSE Markert"].append(numpy.sqrt(markert_sse / no_measurements))
        measurement_output["RMSE Brakelmann"].append(numpy.sqrt(brakelmann_sse / no_measurements))
        measurement_output["RMSE Markle"].append(numpy.sqrt(markle_sse / no_measurements))
        measurement_output["RMSE Hu"].append(numpy.sqrt(hu_sse / no_measurements))
        measurement_output["RMSE Markert spez. u."].append(numpy.sqrt(markert_specific_unpacked_sse / no_measurements))
        measurement_output["RMSE Markert spez. p."].append(numpy.sqrt(markert_specific_packed_sse / no_measurements))
        measurement_output["RMSE Markert Lu"].append(numpy.sqrt(markert_lu_sse / no_measurements))

        markert_avrg_dir = numpy.sum(lambda_measurement - lambda_markert_ideal) / no_measurements
        brakelmann_avrg_dir = numpy.sum(lambda_measurement - lambda_brakelmann_ideal) / no_measurements
        markle_avrg_dir = numpy.sum(lambda_measurement - lambda_markle_ideal) / no_measurements
        hu_avrg_dir = numpy.sum(lambda_measurement - lambda_hu_ideal) / no_measurements
        markert_specific_unpacked_avrg_dir = numpy.sum(lambda_measurement - lambda_markert_specific_unpacked) / no_measurements
        markert_specific_packed_avrg_dir = numpy.sum(lambda_measurement - lambda_markert_specific_packed) / no_measurements
        markert_lu_avrg_dir = numpy.sum(lambda_measurement - lambda_markert_lu) / no_measurements
        measurement_output["DIR Markert"].append(markert_avrg_dir)
        measurement_output["DIR Brakelmann"].append(brakelmann_avrg_dir)
        measurement_output["DIR Markle"].append(markle_avrg_dir)
        measurement_output["DIR Hu"].append(hu_avrg_dir)
        measurement_output["DIR Markert spez. u."].append(markert_specific_unpacked_avrg_dir)
        measurement_output["DIR Markert spez. p."].append(markert_specific_packed_avrg_dir)
        measurement_output["DIR Markert Lu"].append(markert_lu_avrg_dir)

        measurement_type_sequence = ["punctual" in measurement_type.lower()] * no_measurements
        soil_type_sequence = ["t/u" in soil_type.lower()] * no_measurements

        scatter_data["Markert"]["model"].extend(lambda_markert_ideal)
        scatter_data["Markert"]["data"].extend(lambda_measurement)
        scatter_data["Markert"]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Markert"]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Markert"]["is_tu"].extend(soil_type_sequence)

        scatter_data["Brakelmann"]["model"].extend(lambda_brakelmann_ideal)
        scatter_data["Brakelmann"]["data"].extend(lambda_measurement)
        scatter_data["Brakelmann"]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Brakelmann"]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Brakelmann"]["is_tu"].extend(soil_type_sequence)

        scatter_data["Markle"]["model"].extend(lambda_markle_ideal)
        scatter_data["Markle"]["data"].extend(lambda_measurement)
        scatter_data["Markle"]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Markle"]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Markle"]["is_tu"].extend(soil_type_sequence)

        scatter_data["Hu"]["model"].extend(lambda_hu_ideal)
        scatter_data["Hu"]["data"].extend(lambda_measurement)
        scatter_data["Hu"]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Hu"]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Hu"]["is_tu"].extend(soil_type_sequence)

        scatter_data["Markert spez. u."]["model"].extend(lambda_markert_specific_unpacked)
        scatter_data["Markert spez. u."]["data"].extend(lambda_measurement)
        scatter_data["Markert spez. u."]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Markert spez. u."]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Markert spez. u."]["is_tu"].extend(soil_type_sequence)

        scatter_data["Markert spez. p."]["model"].extend(lambda_markert_specific_packed)
        scatter_data["Markert spez. p."]["data"].extend(lambda_measurement)
        scatter_data["Markert spez. p."]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Markert spez. p."]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Markert spez. p."]["is_tu"].extend(soil_type_sequence)

        scatter_data["Markert Lu"]["model"].extend(lambda_markert_lu)
        scatter_data["Markert Lu"]["data"].extend(lambda_measurement)
        scatter_data["Markert Lu"]["is_punctual"].extend(measurement_type_sequence)
        scatter_data["Markert Lu"]["is_in_range"].extend((theta_array >= bound_lo) & (bound_hi >= theta_array))
        scatter_data["Markert Lu"]["is_tu"].extend(soil_type_sequence)

        # Modelle
        lambda_markert = markert_all(
            theta_range,
            percentage_clay,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si)

        lambda_brakelmann = brakelmann(
            theta_range,
            percentage_clay,
            percentage_sand,
            percentage_silt,
            thermal_conductivity_water,
            porosity_ratio)

        lambda_markle = markle(
            theta_range,
            thermal_conductivity_sand,
            porosity_ratio,
            thermal_conductivity_water,
            particle_density,
            density_soil)

        lambda_hu = hu(
            theta_range,
            thermal_conductivity_sand,
            thermal_conductivity_water,
            porosity_ratio,
            particle_density,
            density_soil)

        lambda_markert_specific_unpacked = markert_specific(
            theta_range,
            percentage_clay,
            percentage_silt,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si,
            packed=False)

        lambda_markert_specific_packed = markert_specific(
            theta_range,
            percentage_clay,
            percentage_silt,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si,
            packed=True)

        lambda_markert_lu = markert_lu(
            theta_range,
            percentage_clay,
            percentage_sand,
            porosity_ratio,
            density_soil_non_si)

        # plot
        """
        pyplot.figure()
        pyplot.title(short_name)
        pyplot.plot(theta_range, lambda_markert, c=cmap(0), label="Markert")
        pyplot.plot(theta_range, lambda_markle, c=cmap(1), label="Markle")
        pyplot.plot(theta_range, lambda_brakelmann, c=cmap(2), label="Brakelmann")
        pyplot.plot(theta_range, lambda_hu, c=cmap(3), label="Hu")
        pyplot.plot(theta_range, lambda_markert_specific_unpacked, c=cmap(4), label="Markert spezifisch unpacked")
        pyplot.plot(theta_range, lambda_markert_specific_packed, c=cmap(5), label="Markert spezifisch packed")
        pyplot.plot(theta_range, lambda_markert_lu, c=cmap(6), label="Markert-Lu")
        pyplot.xlabel("Theta [m³%]")
        pyplot.ylabel("Lambda [W/(mK)]")
        pyplot.legend()
        """
        # write to file
        soil_output = {
            f"Feuchte {row_index + 1:d}, {short_name:s} [m³%]":                         theta_range,
            f"Markert {row_index + 1:d}, {short_name:s} [W/(mK)]":                      lambda_markert,
            f"Brakelmann {row_index + 1:d}, {short_name:s} [W/(mK)]":                   lambda_brakelmann,
            f"Markle {row_index + 1:d}, {short_name:s} [W/(mK)]":                       lambda_markle,
            f"Hu {row_index + 1:d}, {short_name:s} [W/(mK)]":                           lambda_hu,
            f"Markert spezifisch unpacked {row_index + 1:d}, {short_name:s} [W/(mK)]":  lambda_markert_specific_unpacked,
            f"Markert spezifisch packed {row_index + 1:d}, {short_name:s} [W/(mK)]":    lambda_markert_specific_packed,
            f"Markert-Lu {row_index + 1:d}, {short_name:s} [W/(mK)]":                   lambda_markert_lu,
        }

        soil_df = pandas.DataFrame(soil_output)
        soil_df.to_excel(soils_output_handler, sheet_name=f"{row_index + 1:d} {short_name:s}")

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
        pyplot.plot([0, 4], [0, 4], c="black", linestyle="--", alpha=.3)

        non_punctual_x = [each_x for each_x, each_is_punctual in zip(info["data"], info["is_punctual"]) if not each_is_punctual]
        non_punctual_y = [each_y for each_y, each_is_punctual in zip(info["model"], info["is_punctual"]) if not each_is_punctual]
        pyplot.scatter(non_punctual_x, non_punctual_y, c="blue", alpha=.1, s=.5)

        punctual_x = [each_x for each_x, each_is_punctual in zip(info["data"], info["is_punctual"]) if each_is_punctual]
        punctual_y = [each_y for each_y, each_is_punctual in zip(info["model"], info["is_punctual"]) if each_is_punctual]
        pyplot.scatter(punctual_x, punctual_y, c="black", alpha=.8, s=8, linewidths=1, marker="x")

        pyplot.xlim(0, 4)
        pyplot.ylim(0, 4)
        pyplot.savefig(f"plots/scatter_{method:s}.pdf")

    # write xls
    measurements_df = pandas.DataFrame(measurement_output)
    measurements_df.to_excel(measurements_output_handler, index=False)
    measurements_output_handler.close()

    soils_output_handler.close()

    pyplot.show()


if __name__ == "__main__":
    main()
