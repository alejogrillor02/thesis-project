#!/usr/bin/env python
# Author: alejo.grillor02

# TODO: Reescribir esto.
"""
Genera los conjuntos (Sin Particionar) de entrenamiento y el conjunto de tests.

Usage:
	process.py <input_csv> [output_path]
"""

import pandas as pd
import numpy as np
import yaml
from sys import argv
from os import path, makedirs, environ
from sklearn.preprocessing import MinMaxScaler


def main():

	def normalizeminmax(train_data: pd.DataFrame, norm_data: pd.DataFrame, feature_name: str) -> tuple:
		"""
		Normaliza una columna por MinMax y guarda stats de normalización.

		Args:
			train_data (pd.DataFrame): Columna de train a normalizar.
			norm_data (pd.DataFrame): Tabla con los datos necesarios para normalizar
			feature_name (str): Nombre de la feature a normalizar.

		Returns:
			train_norm (tuple): Una tupla con ambas columnas normalizadas por MinMax
		"""

		feature_stats = norm_data[norm_data['feature'] == feature_name].iloc[0]
		min_val = feature_stats['min']
		max_val = feature_stats['max']

		train_norm = (train_data - min_val) / (max_val - min_val)

		return train_norm

	input_path = argv[1]
	model_index = path.basename(input_path)[:3]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	FEATURES = config['FEATURES']
	RANDOM_STATE = config['RANDOM_STATE']
	DATA_DIR = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'])

	output_dir = path.join(DATA_DIR, "processed")
	makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(input_path)
	norm_stats = []

	# Guardar los stats de normalización
	for feature in FEATURES:
		if feature == "REF_POST":
			# Caso especial para feature compuesto
			col_data = df["ESF2"] + (df["CIL2"] / 2)
		elif feature == "DELTA_REF":
			col_data = df["ESF2"] + (df["CIL2"] / 2) - df["REF_ESP"]
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

	size = len(df)

	# Shuffle
	df = df.sample(n=size, random_state=RANDOM_STATE)

	data = np.empty((len(df), len(FEATURES) + 1))
	data[:, 0] = [1 if entry == "M" else 0 for entry in df["SEXO"]]

	for i, feature in enumerate(FEATURES, start=1):
		if feature == "REF_POST":
			# Caso especial para feature compuesto
			col_data = df["ESF2"] + (df["CIL2"] / 2)
		elif feature == "DELTA_REF":
			col_data = df["ESF2"] + (df["CIL2"] / 2) - df["REF_ESP"]
		else:
			col_data = df[feature]

		data[:, i] = normalizeminmax(col_data, norm_data, feature)

	output_path = path.join(output_dir, f'{model_index}.txt')
	np.savetxt(output_path, data, fmt="%d " + " ".join(["%f"] * len(FEATURES)))


if __name__ == "__main__":
	main()
