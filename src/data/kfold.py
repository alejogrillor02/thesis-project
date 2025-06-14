#!/usr/bin/env python

"""
Genera folds de entrenamiento usando Kfold.

Usage:
	kfold.py <input_dataset> [output_path]
"""

import numpy as np
import yaml
from sys import argv
from os import path, makedirs, environ
from sklearn.model_selection import StratifiedKFold, KFold


def main():

	input_dataset = argv[1]
	model_index = path.basename(input_dataset)[:3]
	set_index = path.basename(input_dataset)[4]

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']
	FEATURES = config['FEATURES']
	RANDOM_STATE = config['RANDOM_STATE']
	DATA_DIR = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'])

	output_path_base = path.join(DATA_DIR, "train/model_{model_index}/set_{set_index}")
	makedirs(output_path_base, exist_ok=True)
	
	data = np.loadtxt(input_dataset)

	X = data[:, 1:]

	# Configurar el tipo de KFold según el set
	if set_index == "G":
		# Separar la clase que hay que balancear del resto
		sex = data[:, 0].astype(int)
		kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
		split_generator = kf.split(X, sex)
	else:
		kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
		split_generator = kf.split(X)

	# Generar los folds de entrenamiento
	for fold_idx, (train_idx, test_idx) in enumerate(split_generator, 1):
		# train_data = np.column_stack((sex[train_idx], X[train_idx]))
		train_data = X[train_idx]

		np.savetxt(path.join(output_path_base, f"{model_index}_{set_index}_fold_{fold_idx}.txt"), train_data, " ".join(["%f"] * len(FEATURES)))


if __name__ == "__main__":
	main()
