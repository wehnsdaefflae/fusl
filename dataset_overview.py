# !/usr/bin/env python3
# coding=utf-8
from pathlib import Path

import numpy
import pandas
import seaborn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager

from matplotlib import pyplot

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


def dataset_info() -> None:
    input_path = Path("data/")
    output_path = Path("output/")

    measurements_input_file = input_path / "Messdatenbank_FAU_Stand_2023-12-19.xlsx"

    particle_density = 2650  # reindichte stein? soil? grauwacke? https://www.chemie.de/lexikon/Gesteinsdichte.html

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    data_density = "low", "high"
    # data_density = "all",

    for data in data_density:
        all_lambda = list()
        all_theta = list()
        all_density = list()
        all_s = list()
        all_clay = list()
        all_silt = list()
        all_sand = list()

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
            steps = theta_measurement_volumetric / porosity_ratio
            steps_max = numpy.max(steps)
            if 1. < steps_max:
                steps /= steps_max

            if len(theta_measurement_volumetric) < 1 or len(data_measured) < 1:
                print(f"Skipping {row_index + 1:d} due to missing data")
                continue

            for each_theta, each_lambda, each_s in zip(theta_measurement_volumetric, data_measured, steps):
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


def dataset_plots() -> None:
    pyplot.rcParams.update(
        {
            "font.family": "Palatino Linotype",
            # "font.size": 16,
        }
    )
    input_path = Path("data/")
    output_path = Path("output_plots/")
    output_path.mkdir(exist_ok=True)

    measurements_input_file = input_path / "Messdatenbank_FAU_Stand_2023-12-19.xlsx"

    data_measurement_sheets = pandas.read_excel(measurements_input_file, sheet_name=None)
    overview_sheet = data_measurement_sheets.get("Übersicht")
    data_density = "low", "high"
    # data_density = "all",

    particle_density = 2_650

    font_manager = matplotlib.font_manager.fontManager
    font_manager.addfont("/home/mark/Downloads/Palatino Linotype.ttf")

    # seaborn.set(font="Palatino Linotype")

    fig_sand = pyplot.figure(figsize=(3, 3))
    ax_sand = fig_sand.add_subplot(111)  # , aspect="equal")
    ax_sand.set(
        title="sand",
        xlabel="volumetric water content θ [cm3/cm3]",
        ylabel="thermal conductivity λ [W/(m∙K)]"
    )

    fig_no_sand = pyplot.figure(figsize=(3, 3))
    ax_no_sand = fig_no_sand.add_subplot(111)  # , aspect="equal")
    ax_no_sand.set(
        title="no sand",
        xlabel="volumetric water content θ [cm3/cm3]",
        ylabel="thermal conductivity λ [W/(m∙K)]"
    )

    for data in data_density:
        all_lambda_sand = list()
        all_theta_sand = list()

        all_lambda_no_sand = list()
        all_theta_no_sand = list()

        for row_index, (n, row) in enumerate(overview_sheet.iterrows()):
            print(f"{data} {row_index}")

            sheet_index = row.values[0]
            percentage_sand = row.values[2]
            percentage_silt = row.values[3]
            percentage_clay = row.values[4]
            density_soil_non_si = row.values[5]

            each_density = row.values[9]

            if data == "low" and each_density != "low":
                continue
            if data == "high" and each_density != "high":
                continue

            each_sheet = data_measurement_sheets.get(f"{sheet_index}")
            theta_array = each_sheet["θ [cm3/cm3]"].to_numpy()

            lambda_array = each_sheet["λ [W/(m∙K)]"].to_numpy()
            filter_array = (
                    numpy.isfinite(lambda_array)
                    & (0 < theta_array)
            )

            density_soil = density_soil_non_si * 1_000.  # g/cm3 -> kg/m3
            porosity_ratio = 1. - density_soil / particle_density

            theta_measurement_volumetric = theta_array[filter_array]
            data_measured = lambda_array[filter_array]
            steps = theta_measurement_volumetric / porosity_ratio
            steps_max = numpy.max(steps)
            if 1. < steps_max:
                steps /= steps_max

            # plot lambda against theta
            if 50. < percentage_sand:
                ax = fig_sand.gca()
                color = "sienna"
                all_lambda = all_lambda_sand
                all_theta = all_theta_sand
            else:
                ax = fig_no_sand.gca()
                color = "darkred"
                all_lambda = all_lambda_no_sand
                all_theta = all_theta_no_sand

            if data == "low":
                seaborn.lineplot(
                    x=theta_measurement_volumetric, y=data_measured, color=color, alpha=1., ax=ax, size=1.,
                    legend=False
                )
            elif data == "high":
                all_lambda.extend(data_measured)
                all_theta.extend(theta_measurement_volumetric)
            else:
                pass

    cmap_sand = LinearSegmentedColormap.from_list("", [(1, 1, 1, 0), (244/255, 164/255, 96/255, 1)])  # RGBA for sandybrown
    cmap_no_sand = LinearSegmentedColormap.from_list("", [(1, 1, 1, 0), (139/255, 0, 0, 1)])  # RGBA for darkred

    seaborn.kdeplot(
        x=all_theta_sand, y=all_lambda_sand, fill=True, cmap=cmap_sand, alpha=1., bw_adjust=1, ax=ax_sand,
        legend=False
    )
    seaborn.kdeplot(
        x=all_theta_no_sand, y=all_lambda_no_sand, fill=True, cmap=cmap_no_sand, alpha=1., bw_adjust=1, ax=ax_no_sand,
        legend=False
    )

    fig_sand.savefig(output_path / f"sand.pdf", bbox_inches="tight")
    fig_no_sand.savefig(output_path / f"no_sand.pdf", bbox_inches="tight")

    # fig.show()
    pyplot.show()
    # pyplot.clf()


def main():
    # dataset_info()
    dataset_plots()


if __name__ == "__main__":
    main()
