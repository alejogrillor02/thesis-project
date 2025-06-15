#!/usr/bin/env python

"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import shap
import yaml
# import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from os import makedirs, path, environ
from sys import argv


def main():

	def load_fold_data(fold_path, fold_number):
		"""Load fold data from text file"""

		filename = f"{model_index}_fold_{fold_number}.txt"
		filepath = path.join(fold_path, filename)
		data = np.loadtxt(filepath)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y

	# Parse command line arguments
	model_index = argv[1]
	set_index = argv[2]
	train_fold = argv[3]
	test_fold = "2" if train_fold == "1" else "1"

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	# FEATURES = config['FEATURES']
	MODEL_DIR = path.join(environ['PROJECT_ROOT'], config['MODEL_DIR'], f"model_{model_index}/set_{set_index}")
	TRAIN_DATA_DIR = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f"train/model_{model_index}")

	output_path_base = path.join(environ['PROJECT_ROOT'], config['OUTPUT_DIR'], f"model_{model_index}/set_{set_index}")
	makedirs(output_path_base, exist_ok=True)

	# Load models
	model_path = path.join(MODEL_DIR, f'{model_index}_{set_index}_fold_{train_fold}.keras')
	model = load_model(model_path)

	X_test, _ = load_fold_data(TRAIN_DATA_DIR, test_fold)

	if set_index == "E":
		# For embedded model, split into categorical and numerical features
		background = X_test[np.random.choice(X_test.shape[0], 400, replace=False)]
		background_cat = background[:, 0].astype(int)
		background_num = background[:, 1:]
		
		# Compute SHAP values for embedded model
		explainer = shap.GradientExplainer(model, [background_cat, background_num])
		test_cat = X_test[:, 0].astype(int)
		test_num = X_test[:, 1:]
		shap_values = explainer([test_cat, test_num])
	else:
		# For non-embedded model, remove first column if not set E
		X_test = np.delete(X_test, 0, axis=1)
		background = X_test[np.random.choice(X_test.shape[0], 400, replace=False)]
		
		# Compute SHAP values for regular model
		explainer = shap.GradientExplainer(model, background)
		shap_values = explainer(X_test)

	# Gráfico de importancia bruta
	plt.figure(figsize=(12, 8))
	shap.plots.bar(shap_values)
	plt.xlabel('Valor SHAP promedio', fontsize=12)
	plt.ylabel('Feature', fontsize=12)
	plt.title('Impacto de Features en la Predicción (SHAP Values)', fontsize=14)
	plt.tight_layout()
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_fold_{train_fold}_shap_values.pdf'))
	plt.close()

	# # Convertir la lista de arrays en un array 3D (folds, samples, features)
	# shap_values_array = np.array(shap_values_per_fold)

	# # Calcular la media de SHAP para cada feature (conservando el signo)
	# mean_shap = np.mean(shap_values_array, axis=(0, 1)).flatten()

	# # Crear DataFrame
	# importance_df = pd.DataFrame({
	# 	'Feature': FEATURES[:-1],
	# 	'SHAP_mean': mean_shap
	# })

	# plt.figure(figsize=(12, 8))
	# plt.barh(
	# 	importance_df['Feature'], importance_df['SHAP_mean'],
	# 	color=np.where(importance_df['SHAP_mean'] > 0, 'skyblue', 'salmon')
	# )
	# plt.xlabel('Valor SHAP promedio', fontsize=12)
	# plt.ylabel('Feature', fontsize=12)
	# plt.title('Impacto de Features en la Predicción (SHAP Values)', fontsize=14)
	# plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
	# plt.gca().invert_yaxis()
	# plt.tight_layout()
	# plot_path_signed = f"{output_path_base}/feature_impact_shap.pdf"
	# plt.savefig(plot_path_signed)
	# plt.close()


if __name__ == "__main__":
	main()
