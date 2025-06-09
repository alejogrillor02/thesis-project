#!/usr/bin/env python

# TODO: Reescribir esto.
"""
Genera folds de entrenamiento.

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

	output_path = path.join(DATA_DIR, "train")
	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	makedirs(output_path_base, exist_ok=True)
	
	data = np.loadtxt(input_dataset)

	# Separar características y etiquetas (Solo es util para G?)
	X = data[:, 1:]
	y = data[:, 0].astype(int)

	# Configurar el tipo de KFold según el set
	if set_index == "G":
		kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
		split_generator = kf.split(X, y)
	else:
		kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
		split_generator = kf.split(X)

	# Generar los folds de entrenamiento
	for fold_idx, (train_idx, test_idx) in enumerate(split_generator, 1):
		train_data = np.column_stack((y[train_idx], X[train_idx]))

		np.savetxt(f"{output_path_base}/{model_index}_{set_index}_fold_{fold_idx}.txt", train_data, fmt="%d " + " ".join(["%f"] * len(FEATURES)))


if __name__ == "__main__":
	main()
