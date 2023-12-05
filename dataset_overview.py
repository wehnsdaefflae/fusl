# !/usr/bin/env python3
# coding=utf-8
from pathlib import Path

import numpy
import pandas

numpy.seterr(all='raise')


def get_all_values(data: list[float]) -> dict[str, float]:
    return {
        "minimum": numpy.min(data),
        "quantile_1": numpy.quantile(data, .25),
        "median": numpy.median(data),
        "mean": numpy.mean(data),
        "quantile_3": numpy.quantile(data, .75),
        "maximum": numpy.max(data),
    }


def print_values(data: dict[str, float]) -> None:
    print(f"minimum\t{data['minimum']:.5f}")
    print(f"quantile_1\t{data['quantile_1']:.5f}")
    print(f"median\t{data['median']:.5f}")
    print(f"mean\t{data['mean']:.5f}")
    print(f"quantile_3\t{data['quantile_3']:.5f}")
    print(f"maximum\t{data['maximum']:.5f}")


def main() -> None:
    input_path = Path("data/")
    output_path = Path("output/")

    measurements_input_file = input_path / "Messdatenbank_FAU_Stand_2023-11-08.xlsx"

    particle_density = 2650  # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    data_density = "low", "high"

    for data in data_density:
        all_lambda = list()
        all_theta = list()
        all_density = list()
        all_s = list()
        all_clay = list()
        all_silt = list()
        all_sand = list()

        combination_str = f"data-{data}"

        subset_path = output_path / combination_str
        subset_path.mkdir(parents=True, exist_ok=True)

        plot_subset_path = subset_path / "plots"
        plot_subset_path.mkdir(parents=True, exist_ok=True)

        for row_index, (n, row) in enumerate(overview_sheet.iterrows()):
            row_index = int(row_index)

            sheet_index, short_name, percentage_sand, percentage_silt, percentage_clay, density_soil_non_si, soil_type = row.values[:7]

            each_density = row.values[9]

            if data == "low" and each_density != "low":
                continue
            if data == "high" and each_density != "high":
                continue

            density_soil = density_soil_non_si * 1000.  # g/cm3 -> kg/m3
            porosity_ratio = 1. - density_soil / particle_density
            percentage_sand = 0. if isinstance(percentage_sand, str) else percentage_sand
            percentage_silt = 0. if isinstance(percentage_silt, str) else percentage_silt
            percentage_clay = 0. if isinstance(percentage_clay, str) else percentage_clay
            print(f"{row_index + 1:d} \t fSand={percentage_sand:.0f}, fSilt={percentage_silt:.0f}, fClay={percentage_clay:.0f}")

            # volumetrischer Sättigungswassergehalt [m3/m3]
            print(porosity_ratio)

            each_sheet = data_measurement_sheets.get(f"{sheet_index}")
            theta_array = each_sheet["θ [cm3/cm3]"].to_numpy()

            lambda_array = each_sheet["λ [W/(m∙K)]"].to_numpy()
            filter_array = (
                    numpy.isfinite(lambda_array)
                    & (0 < theta_array)
            )

            theta_measurement_volumetric = theta_array[filter_array]
            data_measured = lambda_array[filter_array]
            s = theta_measurement_volumetric / porosity_ratio

            if len(theta_measurement_volumetric) < 1 or len(data_measured) < 1:
                print(f"Skipping {row_index + 1:d} due to missing data")
                continue

            for each_theta, each_lambda, each_s in zip(theta_measurement_volumetric, data_measured, s):
                all_theta.append(each_theta)
                all_lambda.append(each_lambda)
                all_s.append(each_s)

                all_density.append(density_soil_non_si)
                all_clay.append(percentage_clay)
                all_silt.append(percentage_silt)
                all_sand.append(percentage_sand)

        lambda_values = get_all_values(all_lambda)
        theta_values = get_all_values(all_theta)
        s_values = get_all_values(all_s)
        density_values = get_all_values(all_density)
        clay_values = get_all_values(all_clay)
        silt_values = get_all_values(all_silt)
        sand_values = get_all_values(all_sand)

        print(f"data type {data}")
        print(f"lambda ({len(all_lambda):d})")
        print_values(lambda_values)
        print(f"theta ({len(all_theta):d})")
        print_values(theta_values)
        print(f"s ({len(all_s):d})")
        print_values(s_values)
        print(f"density ({len(all_density):d})")
        print_values(density_values)
        print(f"clay ({len(all_clay):d})")
        print_values(clay_values)
        print(f"silt ({len(all_silt):d})")
        print_values(silt_values)
        print(f"sand ({len(all_sand):d})")
        print_values(sand_values)


if __name__ == "__main__":
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
