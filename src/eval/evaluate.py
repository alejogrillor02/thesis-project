#!/usr/bin/env python

"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from os import makedirs, path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

	def load_test_data(test_path):
		"""Load test data from file."""
		data = np.loadtxt(test_path)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y

	def denormalizeminmax(norm_data: np.array, norm_stats: pd.DataFrame) -> np.array:
		"""
		Denormaliza una columna usando las estadísticas guardadas de MinMax.

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
	n_folds = int(argv[3])
	denorm_path = argv[4]
	output_path = argv[5]

	# Parse model and set index
	parts = models_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	makedirs(output_path_base, exist_ok=True)

	# Load test data
	X_test, y_test = load_test_data(test_data_path)
	X_test = np.delete(X_test, 0, axis=1)

	# store metrics for each fold
	all_mae = []
	all_mse = []
	all_r2 = []
	all_predictions = []

	for fold_num in range(1, n_folds + 1):
		model_path = path.join(models_path, f'{model_index}_{set_index}_fold_{fold_num}.keras')

		model = load_model(model_path)

		# Make predictions
		y_pred = model.predict(X_test).flatten()
		all_predictions.append(y_pred)

		# Denormalize labels
		all_stats = pd.read_csv(f'{denorm_path}/{model_index}_norm_stats.csv')
		norm_stats = all_stats.iloc[-1]

		y_test = denormalizeminmax(y_test, norm_stats)
		y_pred = denormalizeminmax(y_pred, norm_stats)

		# Calculate metrics
		mae_score = mean_absolute_error(y_test, y_pred)
		mse_score = mean_squared_error(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)

		all_mae.append(mae_score)
		all_mse.append(mse_score)
		all_r2.append(r2)

		print(f"Fold {fold_num} - MAE: {mae_score:.4f}, MSE: {mse_score:.4f}, R²: {r2:.4f}")

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

	# Optionally: Calculate metrics on mean predictions
	mean_predictions = np.mean(np.array(all_predictions), axis=0)
	ensemble_mae = mean_absolute_error(y_test, mean_predictions)
	ensemble_mse = mean_squared_error(y_test, mean_predictions)
	ensemble_r2 = r2_score(y_test, mean_predictions)

	print("\nEnsemble Performance (Mean Prediction):")
	print(f"MAE: {ensemble_mae:.4f}")
	print(f"MSE: {ensemble_mse:.4f}")
	print(f"R2: {ensemble_r2:.4f}")

	# Plot actual vs predicted
	plt.figure(figsize=(8, 6))
	plt.scatter(y_test, mean_predictions, alpha=0.5)
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
	plt.xlabel('Actual Values')
	plt.ylabel('Predicted Values')
	plt.title('Actual vs Predicted Values')
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_test_predictions.pdf'))


if __name__ == "__main__":
	main()
