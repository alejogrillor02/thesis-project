#!/usr/bin/env python

import numpy as np
import shap
from tensorflow.keras.models import load_model
import sys
from os import path


def main():

	def load_fold_data(fold_path, fold_number):
		"""Docstring..."""

		filename = f"{model_index}_{set_index}_fold_{fold_number}.txt"
		filepath = path.join(fold_path, filename)
		data = np.loadtxt(filepath)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y

	# Parsear argumentos de línea de comandos
	model_path = sys.argv[1]
	training_set_path = sys.argv[2]
	n_folds = int(sys.argv[3])

	parts = model_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	model_paths = [f"{model_path}/{model_index}_{set_index}_fold_{i}.keras" for i in range(1, n_folds + 1)]
	models = [load_model(path) for path in model_paths]

	# Cargar el set de training entero
	X_train_parts = []

	for fold_num in range(1, n_folds + 1):
		X, _y = load_fold_data(training_set_path, fold_num)
		X_train_parts.append(X)

	X_train = np.delete(np.concatenate(X_train_parts, axis=0), 0, axis=1)

	# Get a random sample
	background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]  # Reference dataset

	data = np.loadtxt(f'{training_set_path}/{model_index}_{set_index}_test.txt')
	X_test = data[:, :-1]

	shap_values_per_fold = []
	for model in models:
		explainer = shap.GradientExplainer(model, background)
		shap_values = explainer.shap_values(X_test)
		shap_values_per_fold.append(shap_values)

	# Stack and average SHAP values across folds
	shap_values_aggregated = np.mean(shap_values_per_fold, axis=0)

	shap.summary_plot(shap_values_aggregated, X_test)


if __name__ == "__main__":
	main()
