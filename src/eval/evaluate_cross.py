#!/usr/bin/env python

"""
Script de evaluación para modelos de aprendizaje automático con validación cruzada.

Uso:
	python script.py <models_path> <test_data_path> <denorm_path> <output_path>

Argumentos:
	models_path: Ruta al directorio que contiene los modelos entrenados
	test_data_path: Ruta al archivo con datos de prueba
	denorm_path: Ruta al directorio con estadísticas para desnormalización
	output_path: Ruta donde guardar resultados y gráficos
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
	models_path = argv[1]
	test_data_path = argv[2]
	denorm_path = argv[3]
	output_path = argv[4]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']

	# Parse model and set index
	parts = models_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}/predictions"
	makedirs(output_path_base, exist_ok=True)

	# Load test data
	X_test, y_test = load_test_data(test_data_path)
	X_test = np.delete(X_test, 0, axis=1)

	all_stats = pd.read_csv(f'{denorm_path}/{model_index}_norm_stats.csv')
	norm_stats = all_stats.iloc[-1]

	y_test = denormalizeminmax(y_test, norm_stats)

	# store metrics for each fold
	all_mae = []
	all_mse = []
	all_r2 = []
	all_predictions = []

	for fold_num in range(1, N_FOLDS + 1):
		model_path = path.join(
			models_path, f'{model_index}_{set_index}_fold_{fold_num}.keras')

		model = load_model(model_path)

		# Make predictions
		y_pred = model.predict(X_test).flatten()
		y_pred = denormalizeminmax(y_pred, norm_stats)
		all_predictions.append(y_pred)

		# Calculate metrics
		mae_score = mean_absolute_error(y_test, y_pred)
		mse_score = mean_squared_error(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)

		all_mae.append(mae_score)
		all_mse.append(mse_score)
		all_r2.append(r2)

		print(f"Fold {fold_num} - MAE: {mae_score:.4f}, MSE: {mse_score:.4f}, R²: {r2:.4f}")

		# Plot actual vs predicted
		plt.figure(figsize=(8, 6))
		plt.scatter(y_test, y_pred, alpha=0.5)
		plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

		# Líneas de margen de error
		error_margin = 0.5
		plt.plot(
			[y_test.min(), y_test.max()],
			[y_test.min() - error_margin, y_test.max() - error_margin],
			'r--', lw=1, alpha=0.7
		)
		plt.plot(
			[y_test.min(), y_test.max()],
			[y_test.min() + error_margin, y_test.max() + error_margin],
			'r--', lw=1, alpha=0.7
		)

		plt.xlabel('Actual Values')
		plt.ylabel('Predicted Values')
		plt.title('Actual vs Predicted Values')
		plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_fold_{fold_num}_predictions.pdf'))

	# Compute mean and std of metrics across folds
	mean_mae = np.mean(all_mae)
	std_mae = np.std(all_mae)

	mean_mse = np.mean(all_mse)
	std_mse = np.std(all_mse)

	mean_r2 = np.mean(all_r2)
	std_r2 = np.std(all_r2)

	# Compute metrics on mean predictions
	mean_predictions = np.mean(np.array(all_predictions), axis=0)
	ensemble_mae = mean_absolute_error(y_test, mean_predictions)
	ensemble_mse = mean_squared_error(y_test, mean_predictions)
	ensemble_r2 = r2_score(y_test, mean_predictions)

	print("\nAggregated Performance Across Folds:")
	print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
	print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
	print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")

	print("\nEnsemble Performance (Mean Prediction):")
	print(f"MAE: {ensemble_mae:.4f}")
	print(f"MSE: {ensemble_mse:.4f}")
	print(f"R2: {ensemble_r2:.4f}")

	# Plot actual vs predicted
	plt.figure(figsize=(8, 6))
	plt.scatter(y_test, mean_predictions, alpha=0.)
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

	# Líneas de margen de error
	error_margin = 0.5
	plt.plot(
		[y_test.min(), y_test.max()],
		[y_test.min() - error_margin, y_test.max() - error_margin],
		'r--', lw=1, alpha=0.3
	)
	plt.plot(
		[y_test.min(), y_test.max()],
		[y_test.min() + error_margin, y_test.max() + error_margin],
		'r--', lw=1, alpha=0.3
	)

	plt.xlabel('Actual Values')
	plt.ylabel('Mean Predicted Values')
	plt.title('Actual vs Mean Predicted Values')
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_mean_predictions.pdf'))
	plt.close()


if __name__ == "__main__":
	main()
