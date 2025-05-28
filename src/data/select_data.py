#!/usr/bin/env python
# Author: alejo.grillor02

"""
Genera folds de entrenamiento y EL set de test.

Usage:
	script.py <input_dataset> <n_folds> [output_path=./]
"""

import numpy as np
from sys import argv
from os import path, makedirs
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def main():

	input_dataset = argv[1]
	n_folds = int(argv[2])
	model_index = path.basename(input_dataset)[:3]
	set_index = path.basename(input_dataset)[4]
	features = argv[4:]

	output_path = argv[3]
	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	makedirs(output_path_base, exist_ok=True)

	data = np.loadtxt(input_dataset)

	# Separar características y etiquetas (Solo es util para G?)
	X = data[:, 1:]
	Y = data[:, 0].astype(int)

	# Primero separar conjunto de test (20%) que no se usará en ningún fold
	if set_index == "G":
		X_train, X_test, Y_train, Y_test = train_test_split(
			X, Y, test_size=0.2, stratify=Y, random_state=None)
	else:
		X_train, X_test, Y_train, Y_test = train_test_split(
			X, Y, test_size=0.2, random_state=None)

	# Configurar el tipo de KFold según el set
	if set_index == "G":
		kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)
		split_generator = kf.split(X_train, Y_train)
	else:
		kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
		split_generator = kf.split(X_train)

	# Generar los folds de entrenamiento
	for fold_idx, (train_idx, test_idx) in enumerate(split_generator, 1):
		train_data = np.column_stack((Y_train[train_idx], X_train[train_idx]))

		np.savetxt(f"{output_path_base}/{model_index}_{set_index}_fold_{fold_idx}.txt", train_data, fmt="%d " + " ".join(["%f"] * len(features)))

	test_data = np.column_stack((Y_test, X_test))
	np.savetxt(f"{output_path_base}/{model_index}_{set_index}_test.txt", test_data, fmt="%d " + " ".join(["%f"] * len(features)))


if __name__ == "__main__":
	main()
