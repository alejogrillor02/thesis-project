#!/usr/bin/env python

"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import shap
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from os import makedirs, path, environ
from sys import argv


def main():

	def load_fold_data(fold_path, fold_number):
		"""Load fold data from text file"""

		filename = f"{model_index}_{set_index}_fold_{fold_number}.txt"
		filepath = path.join(fold_path, filename)
		data = np.loadtxt(filepath)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y

	# Parse command line arguments
	model_index = argv[1]
	set_index = argv[2]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']
	FEATURES = config['FEATURES']
	MODEL_DIR = path.join(environ['PROJECT_ROOT'], config['MODEL_DIR'], f"model_{model_index}/set_{set_index}")
	TRAIN_DATA_DIR = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f"train/model_{model_index}/set_{set_index}")

	output_path_base = path.join(environ['PROJECT_ROOT'], config['OUTPUT_DIR'], f"model_{model_index}/set_{set_index}/shap")
	makedirs(output_path_base, exist_ok=True)

	# Load models
	model_paths = [path.join(MODEL_DIR, f'{model_index}_{set_index}_fold_{i}.keras') for i in range(1, N_FOLDS + 1)]
	models = [load_model(path) for path in model_paths]

	data = np.loadtxt(path.join(TRAIN_DATA_DIR, f'{model_index}_{set_index}_test.txt'))
	X_test = data[:, :-1]

	for i in range(N_FOLDS):
		# Load training data
		X_train_parts = []
		for fold in range(1, N_FOLDS + 1):
			if fold == i + 1:
				pass
			else:
				X, _y = load_fold_data(TRAIN_DATA_DIR, fold)
				X_train_parts.append(X)
		X_train = np.concatenate(X_train_parts, axis=0)

		# Get a random sample for background
		background = X_train[np.random.choice(X_train.shape[0], 400, replace=False)]

		explainer = shap.GradientExplainer(models[i], background)
		shap_values = explainer.shap_values(X_test)
		shap_values = np.squeeze(shap_values)  # Aplanar la ultima dimension
		mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

		# Crear DataFrame
		importance_df = pd.DataFrame({
			'Feature': FEATURES[:-1],
			'SHAP_mean': mean_abs_shap
		})

		# Gráfico de importancia bruta
		plt.figure(figsize=(12, 8))
		plt.barh(importance_df['Feature'], importance_df['SHAP_mean'], 'salmon')
		plt.xlabel('Valor SHAP promedio', fontsize=12)
		plt.ylabel('Feature', fontsize=12)
		plt.title('Impacto de Features en la Predicción (SHAP Values)', fontsize=14)
		plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plot_path_signed = f'{output_path_base}/{model_index}_shap_values_fold_{fold}.pdf'
		plt.savefig(plot_path_signed)
		plt.close()


if __name__ == "__main__":
	main()
