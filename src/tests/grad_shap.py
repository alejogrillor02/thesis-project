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
	model_path = argv[1]
	training_set_path = argv[2]
	output_path = argv[3]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']
	FEATURES = config['FEATURES']

	parts = model_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	makedirs(output_path_base, exist_ok=True)

	# Load models
	model_paths = [f"{model_path}/{model_index}_{set_index}_fold_{i}.keras" for i in range(1, N_FOLDS + 1)]
	models = [load_model(path) for path in model_paths]

	# Load training data
	X_train_parts = []
	for fold_num in range(1, N_FOLDS + 1):
		X, _y = load_fold_data(training_set_path, fold_num)
		X_train_parts.append(X)

	X_train = np.delete(np.concatenate(X_train_parts, axis=0), 0, axis=1)

	# Get a random sample for background
	background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

	data = np.loadtxt(f'{training_set_path}/{model_index}_{set_index}_test.txt')
	X_test = data[:, :-1]
	X_test = np.delete(X_test, 0, axis=1)

	# Compute SHAP values for each fold
	shap_values_per_fold = []
	for model in models:
		explainer = shap.GradientExplainer(model, background)
		shap_values = explainer.shap_values(X_test)
		shap_values_per_fold.append(shap_values)

	# # Aggregate SHAP values across folds
	# shap_values_aggregated = np.mean(shap_values_per_fold, axis=0)

	# mean_abs_shap = pd.DataFrame({
	# 	'feature': FEATURES,
	# 	'mean_abs_shap': np.mean(np.abs(shap_values_aggregated), axis=0)
	# }).sort_values('mean_abs_shap', ascending=False)

	# mean_abs_shap.to_csv(path.join(output_path_base, f'{model_index}_{set_index}_shap_feature_importance.csv'), index=False)

	# Convertir la lista de arrays en un array 3D (folds, samples, features)
	shap_values_array = np.array(shap_values_per_fold)

	# Calcular la media de SHAP para cada feature (conservando el signo)
	mean_shap = np.mean(shap_values_array, axis=(0, 1)).flatten()

	# Crear DataFrame
	importance_df = pd.DataFrame({
		'Feature': FEATURES[:-1],
		'SHAP_mean': mean_shap
	})

	# Gráfico de importancia bruta
	plt.figure(figsize=(12, 8))
	plt.barh(
		importance_df['Feature'], importance_df['SHAP_mean'],
		color=np.where(importance_df['SHAP_mean'] > 0, 'skyblue', 'salmon')
	)
	plt.xlabel('Valor SHAP promedio', fontsize=12)
	plt.ylabel('Feature', fontsize=12)
	plt.title('Impacto de Features en la Predicción (SHAP Values)', fontsize=14)
	plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
	plt.gca().invert_yaxis()
	plt.tight_layout()
	plot_path_signed = f"{output_path_base}/feature_impact_shap.pdf"
	plt.savefig(plot_path_signed)
	plt.close()


if __name__ == "__main__":
	main()
