#!/usr/bin/env python

"""
Script de evaluación.

Usage:
	python script.py <model_index> <set_index> <test_data_set_index>

Args:
	model_index (int): Índice del modelo a evaluar
	set_index (int): Índice del conjunto de entrenamiento
	test_data_set_index (int): Índice del conjunto de prueba (opcional, por defecto = set_index)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sys import argv
from os import makedirs, path, environ
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

	def load_test_data(test_path):
		"""
		Carga datos de prueba desde un archivo.

		Args:
			test_path (str): Ruta al archivo de datos de prueba

		Returns:
			tuple: (X_test, y_test) donde X_test son las características y y_test las etiquetas
		"""
		data = np.loadtxt(test_path)
		X = data[:, :-1]
		y = data[:, -1]

		return X, y

	def denormalizeminmax(norm_data: np.array, norm_stats: pd.DataFrame) -> np.array:
		"""
		Desnormaliza una columna usando las estadísticas guardadas de MinMax.

		Args:
				data (np.array): Datos normalizados a desnormalizar
				norm_stats (pd.DataFrame): DataFrame con stats de normalización

		Returns:
				pd.Series: Datos desnormalizados
		"""

		return norm_data * (norm_stats['max'] - norm_stats['min']) + norm_stats['min']

	# Parse command line arguments
	model_index = argv[1]
	set_index = argv[2]
	test_data_set_index = argv[3] if len(argv) > 3 else argv[2]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']
	MODEL_DIR = path.join(environ['PROJECT_ROOT'], config['MODEL_DIR'], f"model_{model_index}/set_{set_index}")

	if set_index == test_data_set_index:
		output_path_base = path.join(environ['PROJECT_ROOT'], config['OUTPUT_DIR'], f"model_{model_index}/set_{set_index}/predictions")
		output_suffix = ""
	else:
		output_path_base = path.join(environ['PROJECT_ROOT'], config['OUTPUT_DIR'], f"model_{model_index}/set_{set_index}/cross_eval/set_{test_data_set_index}")
		output_suffix = f"_with_{test_data_set_index}"
	makedirs(output_path_base, exist_ok=True)

	# Load test data
	test_data_path = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f"train/model_{model_index}/set_{set_index}/{model_index}_{set_index}_test.txt")
	X_test, y_test = load_test_data(test_data_path)
	X_test = np.delete(X_test, 0, axis=1) if set_index != "E" else X_test

	if set_index == "E":
		sex_test = X_test[:, 0].astype(int)
		X_test_num = X_test[:, 1:]

		X_test = {'sex_input': sex_test, 'numerical_input': X_test_num}

	# Load norm metrics
	norm_stats_path = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f'processed/{model_index}_norm_stats.csv')
	all_stats = pd.read_csv(norm_stats_path)
	norm_stats = all_stats.iloc[-1]

	y_test = denormalizeminmax(y_test, norm_stats)

	# store metrics for each fold
	all_mae = []
	all_mse = []
	all_r2 = []
	all_errors = []

	for fold_num in range(1, N_FOLDS + 1):
		model_path = path.join(MODEL_DIR, f'{model_index}_{set_index}_fold_{fold_num}.keras')
		model = load_model(model_path)

		# Make predictions
		y_pred = model.predict(X_test).flatten()
		y_pred = denormalizeminmax(y_pred, norm_stats)
		errors = y_pred - y_test
		indices = np.arange(len(errors))

		bad = len(errors[abs(errors) >= 0.5])
		very_bad = len(errors[abs(errors) >= 1.0])

		# Calculate metrics
		mae_score = mean_absolute_error(y_test, y_pred)
		mse_score = mean_squared_error(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)

		all_mae.append(mae_score)
		all_mse.append(mse_score)
		all_r2.append(r2)
		all_errors.append(errors)

		print(f"Fold {fold_num} - MAE: {mae_score:.4f}, MSE: {mse_score:.4f}, R²: {r2:.4f}, Bad cases: {bad}, Very bad cases: {very_bad}")

		# Plot errors
		plt.figure(figsize=(8, 6))
		plt.scatter(indices, errors, alpha=0.5)
		plt.plot([0, indices.max()], [0, 0], 'k--', lw=2)

		# Líneas de margen de error
		error_margin = 0.5
		plt.plot([0, indices.max()], np.zeros(2) + error_margin, 'r--', lw=1, alpha=0.7)
		plt.plot([0, indices.max()], np.zeros(2) - error_margin, 'r--', lw=1, alpha=0.7)

		plt.ylabel('Errores')
		plt.title('Ploteo de Errores')
		plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}{output_suffix}_fold_{fold_num}_errors.pdf'))

		# Plot actual vs predicted
		# plt.figure(figsize=(8, 6))
		# plt.scatter(y_test, y_pred, alpha=0.5)
		# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
		# # Líneas de margen de error
		# error_margin = 0.5
		# plt.plot(
		# 	[y_test.min(), y_test.max()],
		# 	[y_test.min() - error_margin, y_test.max() - error_margin],
		# 	'r--', lw=1, alpha=0.7
		# )
		# plt.plot(
		# 	[y_test.min(), y_test.max()],
		# 	[y_test.min() + error_margin, y_test.max() + error_margin],
		# 	'r--', lw=1, alpha=0.7
		# )
		# plt.xlabel('Actual Values')
		# plt.ylabel('Predicted Values')
		# plt.title('Actual vs Predicted Values')
		# plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}{output_suffix}_fold_{fold_num}_predictions.pdf'))

	# Guardar métricas por fold en un CSV
	errors_per_fold = pd.DataFrame(all_errors).transpose()

	errors_csv_path = path.join(output_path_base, f'{model_index}_{set_index}{output_suffix}_fold_errors.csv')
	errors_per_fold.to_csv(errors_csv_path, index=False)

	# Compute mean and std of metrics across folds
	mean_mae = np.mean(all_mae)
	std_mae = np.std(all_mae)

	mean_mse = np.mean(all_mse)
	std_mse = np.std(all_mse)

	mean_r2 = np.mean(all_r2)
	std_r2 = np.std(all_r2)

	print("\nAggregated Performance Across Folds:")
	print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
	print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
	print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")


if __name__ == "__main__":
	main()
