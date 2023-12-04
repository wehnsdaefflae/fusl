# !/usr/bin/env python3
# coding=utf-8
from pathlib import Path

import numpy
import pandas

numpy.seterr(all='raise')


def main() -> None:
    input_path = Path("data/")
    output_path = Path("output/")

    measurements_input_file = input_path / "Messdatenbank_FAU_Stand_2023-11-08.xlsx"

    particle_density = 2650  # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    data_density = "low", "high"

    for data in data_density:
        combination_str = f"data-{data}"

        subset_path = output_path / combination_str
        subset_path.mkdir(parents=True, exist_ok=True)

        plot_subset_path = subset_path / "plots"
        plot_subset_path.mkdir(parents=True, exist_ok=True)

        next_sheet = 0

        for row_index, (n, row) in enumerate(overview_sheet.iterrows()):
            row_index = int(row_index)

            while (each_sheet := data_measurement_sheets.get(f"{next_sheet:d}")) is None:
                next_sheet += 1
                if next_sheet >= 100:
                    print(f"Sheet {n + 1:d} not found")
                    break

            short_name, percentage_sand, percentage_silt, percentage_clay, density_soil_non_si, soil_type = row.values[1:7]

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

            theta_array = each_sheet["θ [cm3/cm3]"].to_numpy()

            lambda_array = each_sheet["λ [W/(m∙K)]"].to_numpy()
            filter_array = (
                    numpy.isfinite(lambda_array)
                    & (0 < theta_array)
            )

            theta_measurement_volumetric = theta_array[filter_array]
            data_measured = lambda_array[filter_array]

            if len(theta_measurement_volumetric) < 1 or len(data_measured) < 1:
                print(f"Skipping {row_index + 1:d} due to missing data")
                continue



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
