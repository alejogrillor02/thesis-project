#!/usr/bin/env python
# Author: alejo.grillor02

# TODO: Reescribir esto.
"""
Genera los conjuntos (Sin Particionar) de entrenamiento.

Usage:
	process.py <input_csv> [output_dir=./<prefix>]
"""

import pandas as pd
import numpy as np
import csv
from sys import argv
from os import path, makedirs


def main() -> None:

	def normalizeminmax(data: pd.DataFrame, set_i: str) -> list:
		"""
		Normaliza una columna a rango [0,1] por MinMax y guarda stats de normalización.

		Args:
			data (pd.Series): Columna a normalizar.
			set_i (str): Sufijo para el archivo de stats (F/M/G).

		Returns:
			list: Datos normalizados.
		"""

		cmax = max(data)
		cmin = min(data)
		data_norm = [(x - cmin) / (cmax - cmin) for x in data]

		with open(f"{output_dir}/{index}_{set_i}_norm_stats.csv", "a") as file:
			writer = csv.writer(file)
			writer.writerow([cmax, cmin])
		return data_norm

	input_path = argv[1]
	index = path.basename(input_path)[:3]
	output_dir = argv[2] if len(argv) > 2 else f"./{index}"
	makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(input_path)

	df_f = df[df["SEXO"] == "F"]
	df_m = df[df["SEXO"] == "M"]

	u_size = min(df_f.shape[0], df_m.shape[0])

	# Hace un sample aleatorio de u_size filas
	df_f_sample = df_f.sample(n=u_size)
	df_m_sample = df_m.sample(n=u_size)

	# Hace dos samples aleatorios de ambos sexos para el set de generalizacion
	df_f_sample_gen = df_f.sample(n=u_size // 2)
	df_m_sample_gen = df_m.sample(n=u_size // 2)

	df_gen_sample = pd.concat([df_f_sample_gen, df_m_sample_gen]).sample(frac=1, random_state=None).reset_index(drop=True)

	for df_i, set_i in ((df_f_sample, "F"), (df_m_sample, "M"), (df_gen_sample, "G")):
		try:
			with open(f"{output_dir}/{index}_{set_i}_norm_stats.csv", "w"):
				pass
		except FileNotFoundError:
			pass

		col0 = [1 if entry == "M" else 0 for entry in df["SEXO"]]
		col1 = normalizeminmax(df["REF_ESP"], set_i)
		col2 = normalizeminmax(df["ESF2"] + (df["CIL2"] / 2), set_i)
		col3 = normalizeminmax(df["K1"], set_i)
		col4 = normalizeminmax(df["LAXIAL"], set_i)
		col5 = normalizeminmax(df["PCA"], set_i)
		col6 = normalizeminmax(df["LENTE_DEF"], set_i)

		data = np.empty((len(col0), 7))
		data[:, 0] = col0
		data[:, 1] = col1
		data[:, 2] = col2
		data[:, 3] = col3
		data[:, 4] = col4
		data[:, 5] = col5
		data[:, 6] = col6

		output_path = f"{output_dir}/{index}_{set_i}.txt"
		np.savetxt(output_path, data, fmt="%d %f %f %f %f %f %f")


if __name__ == "__main__":
	main()
