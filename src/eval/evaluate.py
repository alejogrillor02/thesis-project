#!/usr/bin/env python

"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from os import path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

	def load_test_data(test_path):
		"""Load test data from file."""
		data = np.loadtxt(test_path)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y

	# Parse command line arguments
	models_path = argv[1]  # Path where trained models are stored
	test_data_path = argv[2]  # Path to test data file

	# Parse model and set index from path (similar to training script)
	parts = models_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	# Load test data
	X_test, y_test = load_test_data(test_data_path)
	X_test = np.delete(X_test, 0, axis=1)  # Consistent with training preprocessing

	# Initialize lists to store metrics for each fold
	all_mae = []
	all_mse = []
	all_r2 = []
	all_predictions = []

	# Evaluate each fold model
	n_folds = 5  # Adjust based on your actual number of folds
	for fold_num in range(1, n_folds + 1):
		model_path = path.join(models_path, f'{model_index}_{set_index}_fold_{fold_num}.keras')

		try:
			model = load_model(model_path)

			# Make predictions
			y_pred = model.predict(X_test).flatten()
			all_predictions.append(y_pred)

			# Calculate metrics
			mae_score = mean_absolute_error(y_test, y_pred)
			mse_score = mean_squared_error(y_test, y_pred)
			r2 = r2_score(y_test, y_pred)

			all_mae.append(mae_score)
			all_mse.append(mse_score)
			all_r2.append(r2)

			print(f"Fold {fold_num} - MAE: {mae_score:.4f}, MSE: {mse_score:.4f}, R²: {r2:.4f}")

		except Exception as e:
			print(f"Error loading model for fold {fold_num}: {e}")
			continue

	# Calculate mean and std of metrics across folds
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
	print(f"R²: {ensemble_r2:.4f}")

	# Plot actual vs predicted
	plt.figure(figsize=(8, 6))
	plt.scatter(y_test, mean_predictions, alpha=0.5)
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
	plt.xlabel('Actual Values')
	plt.ylabel('Predicted Values')
	plt.title('Actual vs Predicted Values')
	plt.savefig(path.join(models_path, f'{model_index}_{set_index}_test_predictions.pdf'))
	plt.close()


if __name__ == "__main__":
	main()
