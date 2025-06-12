#!/usr/bin/env python

# TODO: Write Docstrings...
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import pandas as pd
import yaml
import numpy as np
from sys import argv
from os import path, environ
from scipy.stats import ttest_rel, shapiro, wilcoxon


def main():

	metric_csv_path_a = argv[1]
	metric_csv_path_b = argv[2]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	NORM_ALPHA = config['NORM_ALPHA']
	ALPHA = config['ALPHA']
	N_FOLDS = config['N_FOLDS']

	# output_path_base = path.join(environ['PROJECT_ROOT'], config['OUTPUT_DIR'], f"model_{model_index}")

	df_a = pd.read_csv(metric_csv_path_a)
	df_b = pd.read_csv(metric_csv_path_b)

	for fold in range(N_FOLDS):
		a_fold_errors = np.array(df_a.iloc[:, fold].tolist())
		b_fold_errors = np.array(df_b.iloc[:, fold].tolist())

		diff = a_fold_errors - b_fold_errors
		shapiro_stat_, p_normality = shapiro(diff)
		print(f'Fold {fold + 1}: {p_normality:.8f}')

		if p_normality > NORM_ALPHA:
			print("Differences are normally distributed; using t-test.")
			t_stat, p_value = ttest_rel(a_fold_errors, b_fold_errors)

			print(f"t-statistic: {t_stat:.8f}")
			print(f"p-value: {p_value:.8f}")
			print(f"Significant at α={ALPHA}? {p_value < ALPHA}")

		else:
			print("Differences not normal; using Wilcoxon test.")
			wilcoxon_stat_, p_value = wilcoxon(a_fold_errors, b_fold_errors)

			print(f"p-value: {p_value:.8f}")
			print(f"Significant at α={ALPHA}? {p_value < ALPHA}")

		print('')


if __name__ == "__main__":
	main()
