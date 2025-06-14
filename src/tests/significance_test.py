#!/usr/bin/env python

# TODO: Write Docstrings...
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import pandas as pd
import yaml
import numpy as np
# import matplotlib.pyplot as plt
from sys import argv
from os import path, environ
from scipy.stats import ttest_rel, shapiro, wilcoxon  # probplot


def main():

	metric_csv_path_a = argv[1]
	metric_csv_path_b = argv[2]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	NORM_ALPHA = config['NORM_ALPHA']
	ALPHA = config['ALPHA']

	# output_path_base = path.join(environ['PROJECT_ROOT'], config['OUTPUT_DIR'], f"model_{model_index}")

	df_a = pd.read_csv(metric_csv_path_a)
	df_b = pd.read_csv(metric_csv_path_b)
	metric = "MAE"

	a_metric = np.array(df_a[metric])
	b_metric = np.array(df_b[metric])

	diff = a_metric - b_metric
	shapiro_stat_, p_normality = shapiro(diff)

	# Create Q-Q plot
	# plt.figure(figsize=(8, 6))
	# probplot(diff, dist="norm", plot=plt)
	# plt.title(f'Q-Q Plot for Fold {fold + 1} Differences')
	# plt.show()

	if p_normality > NORM_ALPHA:  # Can assume errors aren't normally distributed
		print("Differences are normally distributed; using t-test.")
		t_stat, p_value = ttest_rel(a_metric, b_metric)

		print(f"t-statistic: {t_stat:.8f}")
		print(f"p-value: {p_value:.8f}")
		print(f"Significant at α={ALPHA}? {p_value < ALPHA}")

	else:
		print("Differences not normal; using Wilcoxon test.")
		wilcoxon_stat_, p_value = wilcoxon(a_metric, b_metric)

		print(f"p-value: {p_value:.8f}")
		print(f"Significant at α={ALPHA}? {p_value < ALPHA}")


if __name__ == "__main__":
	main()
