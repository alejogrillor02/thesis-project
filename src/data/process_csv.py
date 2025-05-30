#!/usr/bin/env python
# Author: alejo.grillor02

# TODO: Reescribir esto.
"""
Genera los conjuntos (Sin Particionar) de entrenamiento y el conjunto de tests.

Usage:
	process.py <input_csv> [output_dir] <features>
"""

import pandas as pd
import numpy as np
from sys import argv
from os import path, makedirs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def main():

	def normalizeminmax(train_data: pd.DataFrame, test_data: pd.DataFrame, norm_data: pd.DataFrame, feature_name: str) -> tuple:
		"""
		Normaliza una columna por MinMax y guarda stats de normalización.

		Args:
			train_data (pd.DataFrame): Columna de train a normalizar.
			test_data (pd.DataFrame): Columna de test a normalizar.
			norm_data (pd.DataFrame): Tabla con los datos necesarios para normalizar
			feature_name (str): Nombre de la feature a normalizar.

		Returns:
			train_norm, test_norm (tuple): Una tupla con ambas columnas normalizadas por MinMax
		"""

		feature_stats = norm_data[norm_data['feature'] == feature_name].iloc[0]
		min_val = feature_stats['min']
		max_val = feature_stats['max']

		train_norm = (train_data - min_val) / (max_val - min_val)
		test_norm = (test_data - min_val) / (max_val - min_val)

		return train_norm, test_norm

	input_path = argv[1]
	model_index = path.basename(input_path)[:3]
	output_dir = argv[2]
	features = argv[3:]

	makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(input_path)
	norm_stats = []

	# Guardar los stats de normalización
	for feature in features:
		if feature == "ESF2+CIL2/2":
			# Caso especial para feature compuesto
			col_data = df["ESF2"] + (df["CIL2"] / 2)
		else:
			col_data = df[feature]

		# Calcular estadísticas de normalización
		scaler = MinMaxScaler()
		scaler.fit(col_data.values.reshape(-1, 1))  # Ajustar el scaler (pero no transformar)

		norm_stats.append({
			'feature': feature,
			'min': scaler.data_min_[0],
			'max': scaler.data_max_[0],
			'data_range': scaler.data_range_[0]
		})

	norm_data = pd.DataFrame(norm_stats)
	norm_data.to_csv(f"{output_dir}/{model_index}_norm_stats.csv", index=False)

	df_f = df[df["SEXO"] == "F"]
	df_m = df[df["SEXO"] == "M"]

	u_size = min(len(df_f), len(df_m))

	df_f = df_f.sample(n=u_size, random_state=None)
	df_m = df_m.sample(n=u_size, random_state=None)

	# Hace un sample aleatorio de u_size filas
	df_f_train, df_f_test = train_test_split(df_f, test_size=0.2, random_state=None)
	df_m_train, df_m_test = train_test_split(df_m, test_size=0.2, random_state=None)

	# Tamaño uniforme
	u_train_size = min(df_f_train.shape[0], df_m_train.shape[0])
	u_test_size = min(df_f_test.shape[0], df_m_test.shape[0])

	# Hace dos samples aleatorios de ambos sexos para el set general
	df_f_sample_g = df_f_train.sample(n=u_train_size // 2)
	df_m_sample_g = df_m_train.sample(n=u_train_size // 2)
	df_f_test_sample_g = df_f_test.sample(n=u_test_size // 2)
	df_m_test_sample_g = df_m_test.sample(n=u_test_size // 2)

	df_g_train = pd.concat([df_f_sample_g, df_m_sample_g]).sample(frac=1, random_state=None).reset_index(drop=True)
	df_g_test = pd.concat([df_f_test_sample_g, df_m_test_sample_g]).sample(frac=1, random_state=None).reset_index(drop=True)

	for df_i, df_i_test, set_i in ((df_f_train, df_f_test, "F"), (df_m_train, df_m_test, "M"), (df_g_train, df_g_test, "G")):

		data = np.empty((len(df_i), len(features) + 1))
		data[:, 0] = [1 if entry == "M" else 0 for entry in df_i["SEXO"]]

		data_test = np.empty((len(df_i_test), len(features) + 1))
		data_test[:, 0] = [1 if entry == "M" else 0 for entry in df_i_test["SEXO"]]

		for i, feature in enumerate(features, start=1):
			if feature == "ESF2+CIL2/2":
				# Caso especial para feature compuesto
				col_data = df_i["ESF2"] + (df_i["CIL2"] / 2)
				col_test_data = df_i_test["ESF2"] + (df_i_test["CIL2"] / 2)
			else:
				col_data = df_i[feature]
				col_test_data = df_i_test[feature]

			data[:, i], data_test[:, i] = normalizeminmax(col_data, col_test_data, norm_data, feature)

		output_path = f"{output_dir}/{model_index}_{set_i}.txt"
		output_path_test = f"{output_dir}/{model_index}_{set_i}_test.txt"
		np.savetxt(output_path, data, fmt="%d " + " ".join(["%f"] * len(features)))
		np.savetxt(output_path_test, data_test, fmt="%d " + " ".join(["%f"] * len(features)))


if __name__ == "__main__":
	main()
