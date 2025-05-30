#!/usr/bin/env python

import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import pandas as pd
from sys import argv
from os import path


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
	n_folds = int(argv[3])
	output_path = argv[4]
	features = argv[5:]

	parts = model_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	os.makedirs(output_path_base, exist_ok=True)

	# Load models
	model_paths = [f"{model_path}/{model_index}_{set_index}_fold_{i}.keras" for i in range(1, n_folds + 1)]
	models = [load_model(path) for path in model_paths]

	# Load training data
	X_train_parts = []
	for fold_num in range(1, n_folds + 1):
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

	# Aggregate SHAP values across folds
	shap_values_aggregated = np.mean(shap_values_per_fold, axis=0)

	shap_explanation = shap.Explanation(
		values=shap_values_aggregated,
		base_values=np.mean([model.predict(background).mean() for model in models]),  # Average model output
		data=X_test,
		feature_names=features
	)

	# 1. Beeswarm plot
	plt.figure(figsize=(12, 8))
	shap.plots.beeswarm(shap_explanation, show=False)
	plt.title("SHAP Beeswarm Plot", fontsize=14)
	plt.tight_layout()
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_shap_beeswarm.pdf'))
	plt.close()

	# 2. Horizontal bar plot of mean absolute SHAP values
	plt.figure(figsize=(12, 8))
	shap.plots.bar(shap_explanation, show=False)
	plt.title("Feature Importance (Mean Absolute SHAP Value)", fontsize=14)
	plt.tight_layout()
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_shap_bar.pdf'))
	plt.close()

	# 3. Heatmap visualization
	plt.figure(figsize=(12, 8))
	shap.plots.heatmap(shap_explanation, show=False)
	plt.title("SHAP Heatmap", fontsize=14)
	plt.tight_layout()
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_shap_heatmap.pdf'))
	plt.close()

	# # 4. Individual feature plots for top 5 features
	# top_features = np.argsort(np.mean(np.abs(shap_values_aggregated), axis=0))[-5:][::-1]
	# for i, feature_idx in enumerate(top_features):
	#     plt.figure(figsize=(10, 6))
	#     shap.plots.scatter(shap_explanation[:, feature_idx], show=False)
	#     plt.title(f"SHAP Dependency Plot for {features[feature_idx]}", fontsize=12)
	#     plt.tight_layout()
	#     plt.savefig(path.join(output_path_base,
	#                         f'{model_index}_{set_index}_shap_feature_{i+1}_{features[feature_idx]}.png'),
	#                 bbox_inches="tight", dpi=300)
	#     plt.close()

	mean_abs_shap = pd.DataFrame({
		'feature': features,
		'mean_abs_shap': np.mean(np.abs(shap_values_aggregated), axis=0)
	}).sort_values('mean_abs_shap', ascending=False)

	mean_abs_shap.to_csv(path.join(output_path_base, f'{model_index}_{set_index}_shap_feature_importance.csv'), index=False)


if __name__ == "__main__":
	main()
