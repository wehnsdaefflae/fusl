def generate_html_table(cell_values: list[list[str]]) -> str:
    """
    Generates an HTML table with four rows and seven columns.

    Parameters:
    cell_values (list of lists): A 2D list containing the cell values for the table.

    Returns:
    str: HTML code for the table.
    """
    if len(cell_values) != 4 or any(len(row) != 7 for row in cell_values):
        return "Invalid input: cell_values must be a 4x7 matrix."

    html = "<table border='1'>\n"

    # Adding rows to the table
    for row in cell_values:
        html += "<tr>\n"
        for cell in row:
            html += f"<td>{cell}</td>\n"
        html += "</tr>\n"

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

    # Prepare the cell values as per the user's request
    cell_values = [
        ["1", "1", "a", "b", "c", "d", "e", "f"],
        ["2", "2", "", "", "", "", "", ""],
        ["3", "3", "", "", "", "", "", ""],
        ["4", "4", "", "", "", "", "", ""]
    ]

    # Correcting the cell values to match the required 4x7 format
    corrected_cell_values = [row[1:] for row in cell_values]  # Remove one element from each row

    html_table = generate_html_table(corrected_cell_values)
    print(html_table)


if __name__ == '__main__':
    main()
