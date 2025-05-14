#!/usr/bin/env python
# Author: alejo.grillor02

"""
Convierte archivos Excel (.xlsx) a formato CSV.

Usage:
    xslx2csv.py <input.xlsx> [output.csv]

Si no se especifica output.csv, se usa el mismo nombre del archivo de entrada.
"""

import pandas as pd
import sys


def xslx2csv(path: str, output_path: str):
    """
    Convierte un archivo Excel a CSV.

    Args:
        path (str): Ruta del archivo Excel de entrada.
        output_path (str): Ruta donde se guardará el archivo CSV.
    """

    df = pd.read_excel(path)
    df.to_csv(output_path, index=False)


def main():
    xslx_path = sys.argv[1]
    try:
        csv_path = sys.argv[2]
    except IndexError:
        csv_path = sys.argv[1][:-5] + ".csv"

    xslx2csv(xslx_path, csv_path)


if __name__ == "__main__":
    main()
