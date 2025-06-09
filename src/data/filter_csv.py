#!/usr/bin/env python
# Author: alejo.grillor02

"""
Filtra de un CSV datos innecesarios y guarda los resultados.

Usage:
	filter_csv.py <input_csv>
"""

from sys import argv
from os import path, makedirs, environ
import pandas as pd
import yaml


def main():

	def filter_mask(dataframe: pd.DataFrame, filter_func) -> pd.DataFrame:
		"""
		Aplica una función de filtro a un dataframe y devuelve el dataframe filtrado.

		Args:
			dataframe (pd.DataFrame): El dataframe a filtrar.
			filter_func (function): La función de filtro a aplicar. Debe devolver un valor booleano para cada fila.

		Returns:
			pd.DataFrame: El dataframe filtrado, con las mismas columnas que el original.
		"""

		mask = dataframe.apply(filter_func, axis=1)
		filtered_dataframe = dataframe[mask]
		return filtered_dataframe

	csv_path = argv[1]
	basename = path.basename(csv_path)

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	DATA_DIR = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'])
	output_dir = path.join(DATA_DIR, "processed/filtered")
	output_path = path.join(output_dir, f"{basename[:3]}_filtered.csv")
	makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(csv_path)

	def filter1(row) -> bool:
		return not row["COMPLICADO"]

	def filter2(row) -> bool:
		return True if row["CIL1"] >= -3.0 and row["CIL1"] <= 3.0 else False

	def filter3(row) -> bool:
		return True if row["AVCC2"] >= 0.5 else False

	def filter4(row) -> bool:
		return True if row["LAXIAL"] >= 19.36 and row["LAXIAL"] <= 27 else False

	def filter5(row) -> bool:
		return True if row["K1"] >= 36 and row["K1"] <= 50.9 else False

	def filter6(row) -> bool:
		srf = row["ESF2"] + row["CIL2"] / 2 - row["REF_ESP"]
		return True if srf <= 3 else False

	def filter7(row) -> bool:
		return True if row["PCA"] >= 2 and row["PCA"] <= 4 else False

	# Aplica los filtros al dataframe
	df_filtered = filter_mask(df, filter1)
	df_filtered = filter_mask(df_filtered, filter2)
	df_filtered = filter_mask(df_filtered, filter3)
	df_filtered = filter_mask(df_filtered, filter4)
	df_filtered = filter_mask(df_filtered, filter5)
	df_filtered = filter_mask(df_filtered, filter6)
	df_filtered = filter_mask(df_filtered, filter7)

	# Elimina los duplicados por la columna HC
	df_filtered = df_filtered.drop_duplicates(subset=["HC"], keep="first")

	df_filtered.to_csv(output_path, index=False)


if __name__ == "__main__":
	main()
