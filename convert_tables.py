def csv_to_html_table(csv_data: str) -> str:
    # Parse the CSV data
    lines = csv_data.strip().split("\n")
    data = {}
    for line in lines:
        parts = line.split(",")
        key = parts[0].split("_")
        sand = 'S = 51-100 %' if 'high' in key[1] else 'S = 0-50 %'
        water = key[2].replace('low_water', 'W = 0-50 %').replace('high_water', 'W = 51-100%').replace('all_water', 'W = 0-100%')
        data_key = (sand, water)
        data[data_key] = (parts[2], parts[3])

    # HTML table setup
    html = "<table border='1'>\n"
    headers = ["Sand:", "S = 0-50 %", "S = 51-100 %", "S = 0-100 %"]
    rows = ["W = 0-50 %, lo data", "W = 51-100%, lo data", "W = 0-100%, lo data",
            "W = 0-50 %, hi data", "W = 51-100%, hi data", "W = 0-100%, hi data"]

    # Create the header row
    html += "  <tr>\n"
    for header in headers:
        html += f"    <th>{header}</th>\n"
    html += "  </tr>\n"

    # Create the data rows
    for row_label in rows:
        html += "  <tr>\n"
        html += f"    <td>{row_label}</td>\n"
        for sand_col in headers[1:]:
            cell_data = data.get((sand_col, row_label), ("", ""))
            html += f"    <td>({cell_data[0]}, {cell_data[1]})</td>\n"
        html += "  </tr>\n"

    html += "</table>"
    return html


def main() -> None:
    csv_data = """data-low_sand-low_water-low,kersten_johansen_bertermann,1.105,5.909
data-low_sand-low_water-high,kersten_johansen_bertermann,0.509,-0.361
data-low_sand-low_water-all,kersten_johansen_bertermann,0.872,-1.232
data-low_sand-high_water-low,kersten_johansen_bertermann,0.714,-0.054
data-low_sand-high_water-high,kersten_johansen_bertermann,0.482,0.133
data-low_sand-high_water-all,kersten_johansen_bertermann,0.609,0.088
data-low_sand-all_water-low,kersten_johansen_bertermann,1.043,-293.329
data-low_sand-all_water-high,kersten_johansen_bertermann,0.504,-0.222
data-low_sand-all_water-all,kersten_johansen_bertermann,0.827,-0.743
data-high_sand-low_water-low,kersten_johansen_bertermann,1.089,4.668
data-high_sand-low_water-high,kersten_johansen_bertermann,0.446,-0.363
data-high_sand-low_water-all,kersten_johansen_bertermann,0.832,-1.207
data-high_sand-high_water-low,kersten_johansen_bertermann,0.626,-0.379
data-high_sand-high_water-high,kersten_johansen_bertermann,0.211,-0.045
data-high_sand-high_water-all,kersten_johansen_bertermann,0.467,-0.12
data-high_sand-all_water-low,kersten_johansen_bertermann,0.842,-2.067
data-high_sand-all_water-high,kersten_johansen_bertermann,0.326,-0.152
data-high_sand-all_water-all,kersten_johansen_bertermann,0.639,-0.395
"""

    html_table = csv_to_html_table(csv_data)
    print(html_table)


if __name__ == '__main__':
    main()
