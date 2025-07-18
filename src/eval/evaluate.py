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
	test_data_path = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f"train/model_{model_index}/set_{test_data_set_index}/{model_index}_{test_data_set_index}_test.txt")
	X_test, y_test = load_test_data(test_data_path)

	# Load norm metrics
	norm_stats_path = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f'processed/{model_index}_norm_stats.csv')
	all_stats = pd.read_csv(norm_stats_path)
	norm_stats = all_stats.iloc[-1]

	y_test = denormalizeminmax(y_test, norm_stats)

	# Sort them through actual values
	sorted_indices = np.argsort(y_test)

	# store metrics for each fold
	all_mae = []
	all_mse = []
	all_r2 = []
	all_errors = []
	all_bad = []
	all_very_bad = []

	# For plotting later
	bin_size = 2
	bins = np.arange(y_test.min(), y_test.max() + bin_size, bin_size)
	indices = np.arange(len(y_test))
	bin_indices = []
	for bin_val in bins:
		# Find the first y_test >= bin_val (since y_test is sorted)
		idx = np.searchsorted(y_test, bin_val, side='left')
		if idx < len(y_test):  # Ensure we don't go out of bounds
			bin_indices.append(idx)
		else:
			bin_indices.append(len(y_test) - 1)  # Fallback to last index

	for fold_num in range(1, N_FOLDS + 1):
		model_path = path.join(MODEL_DIR, f'{model_index}_{set_index}_fold_{fold_num}.keras')
		model = load_model(model_path)

		# Make predictions
		y_pred = model.predict(X_test).flatten()
		y_pred = denormalizeminmax(y_pred, norm_stats)

		errors = y_pred - y_test
		errors_sorted = errors[sorted_indices]  # Apply the same sorting.

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
		all_bad.append(bad)
		all_very_bad.append(very_bad)

		print(f"Fold {fold_num} - MAE: {mae_score:.4f}, MSE: {mse_score:.4f}, R²: {r2:.4f}, Bad cases: {bad}, Very bad cases: {very_bad}")

		# Plot errors
		plt.figure(figsize=(8, 6))
		plt.scatter(indices, errors_sorted, alpha=0.5)
		plt.axhline(0, color='k', linestyle='--', lw=2, alpha=0.3)

		# Líneas de margen de error
		error_margin = 0.5
		plt.axhline(error_margin, color='r', linestyle='--', lw=1, alpha=0.7)
		plt.axhline(-error_margin, color='r', linestyle='--', lw=1, alpha=0.7)

		# Replace x-axis indices with bins
		x_ticks = bin_indices
		x_labels = [f"{bin_i}" for bin_i in bins]  # Or use bin values directly
		plt.xticks(x_ticks, x_labels, rotation=45)

		# Offset every other label
		for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
			if i == 0:
				prev_pos_down = True
			elif bin_indices[i] - bin_indices[i - 1] <= 50:
				pos = label.get_position()[1]
				new_pos = pos + 1 * 0.08 if prev_pos_down else pos
				prev_pos_down = not prev_pos_down
				label.set_y(new_pos)
			else:
				prev_pos_down = True
		
		plt.gca().tick_params(
			axis='x',
			which='both',
			direction='inout'
		)

		plt.xlabel('Rangos de Valores reales')
		plt.ylabel('Errores de Predicción')
		# plt.title('Ploteo de Errores')
		plt.tight_layout()  # Prevent label cutoff
		plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}{output_suffix}_fold_{fold_num}_errors.pdf'))
		plt.close()

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

	metrics_per_fold = pd.DataFrame({
		'MAE': all_mae,
		'MSE': all_mse,
		'R2': all_r2,
		'Bad': all_bad,
		'Very Bad': all_very_bad
	})

	# Compute mean and std of metrics across folds
	mean_mae = np.mean(all_mae)
	std_mae = np.std(all_mae)

	mean_mse = np.mean(all_mse)
	std_mse = np.std(all_mse)

	mean_r2 = np.mean(all_r2)
	std_r2 = np.std(all_r2)

	metrics_csv_path = path.join(output_path_base, f'{model_index}_{set_index}{output_suffix}_fold_metrics.csv')
	metrics_per_fold.to_csv(metrics_csv_path, index=False)

	print("\nAggregated Performance Across Folds:")
	print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
	print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
	print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")


if __name__ == "__main__":
	main()
