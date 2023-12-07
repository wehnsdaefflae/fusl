import os

import pandas


def main() -> None:
    input_dir = "output"
    output_file = "output/result_overview_new.csv"

    for data_set_name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, data_set_name)
        if not os.path.isdir(full_path) or not data_set_name.startswith("data"):
            continue

        sheet_file = os.path.join(full_path, "model_fit.xlsx")
        sheet_model_fit = pandas.read_excel(sheet_file, sheet_name=0)

        # iterate over all columns
        for n in range(2, len(sheet_model_fit.columns), 2):
            each_rmse_column = sheet_model_fit.columns[n]
            each_bias_column = sheet_model_fit.columns[n + 1]

            avrg_rmse = sheet_model_fit.iloc[:, n].mean()
            avrg_bias = sheet_model_fit.iloc[:, n + 1].mean()

            column_name = each_rmse_column.split(" ")[-1]

            with open(output_file, mode="a") as f:
                f.write(f"{data_set_name}\t{column_name}\t{avrg_rmse}\t{avrg_bias}\n")


if __name__ == "__main__":
    main()
