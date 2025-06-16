#!/usr/bin/env python

# TODO: Write Docstrings... and fix missing vars
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import yaml
from os import makedirs, path, environ
from sys import argv
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mae
from tensorflow.keras.optimizers import Adam


# TODO: This thing does not work frfr
def main():

	def create_model(
		inputs=10, outputs=1,
		hidden_layers=2,
		first_layer_neurons=128,
		layer_reduction=0.5,
		dropout_rate=0.3,
		l2_reg=0.01,
		learning_rate=0.001
	):
		"""Create a configurable MLP model for regression."""
		model = Sequential()
		model.add(Input(shape=(inputs,)))
		
		# Add hidden layers with reducing number of neurons
		neurons = first_layer_neurons
		for _ in range(hidden_layers):
			model.add(Dense(int(neurons), activation=relu, kernel_regularizer=l2(l2_reg)))
			model.add(BatchNormalization())
			model.add(Dropout(dropout_rate))
			neurons *= layer_reduction
		
		model.add(Dense(outputs))
		
		# Compile model
		optimizer = Adam(learning_rate=learning_rate)
		model.compile(loss=mae, optimizer=optimizer, metrics=[mae])
		return model

	def load_fold_data(fold_path, fold_number):
		"""Docstring..."""

		filename = f"{model_index}_{set_index}_fold_{fold_number}.txt"
		filepath = path.join(fold_path, filename)
		data = np.loadtxt(filepath)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y
	
	model_index = argv[1]
	set_index = argv[2]
	val_fold = int(argv[3])

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']
	EPOCHS = config['EPOCHS']
	BATCH_SIZE = config['BATCH_SIZE']

	TRAINDATA_DIR = path.join(environ['PROJECT_ROOT'], config['DATA_DIR'], f"train/model_{model_index}/set_{set_index}")

	output_path = path.join(environ['PROJECT_ROOT'], config['MODEL_DIR'], f"model_{model_index}/set_{set_index}")
	makedirs(output_path, exist_ok=True)

	# Cargar el set de training y de validación
	X_train_parts = []
	y_train_parts = []

	for fold_num in range(1, N_FOLDS + 1):
		if fold_num == val_fold:
			X_val, y_val = load_fold_data(TRAINDATA_DIR, fold_num)
		else:
			X, y = load_fold_data(TRAINDATA_DIR, fold_num)
			X_train_parts.append(X)
			y_train_parts.append(y)

	# Concatenate all training parts
	X_train = np.concatenate(X_train_parts, axis=0)
	y_train = np.concatenate(y_train_parts, axis=0)

	# Get input and output dimensions
	n_inputs = X_train.shape[1]
	n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]

	# Create KerasRegressor wrapper
	model = KerasRegressor(
		model=create_model,
		inputs=n_inputs,
		outputs=n_outputs,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		verbose=1
	)

	# Define the grid search parameters
	param_grid = {
		'hidden_layers': [1, 2, 3],
		'first_layer_neurons': [64, 128, 256],
		'layer_reduction': [0.3, 0.5, 0.7],
		'dropout_rate': [0.2, 0.3, 0.4],
		'l2_reg': [0.001, 0.01, 0.1],
		'learning_rate': [0.0001, 0.001, 0.01]
	}

	# Create GridSearchCV
	grid = GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		cv=3,  # Using 3-fold CV within the training set
		scoring='neg_mean_absolute_error',  # For regression
		n_jobs=-1,
		verbose=2
	)

	# Fit the grid search
	grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))

	# Save the best model
	best_model = grid_result.best_estimator_.model
	best_model.save(path.join(output_path, f'best_model_fold_{val_fold}.h5'))

	# Save grid search results
	with open(path.join(output_path, f'grid_results_fold_{val_fold}.txt'), 'w') as f:
		f.write(f"Best: {grid_result.best_score_} using {grid_result.best_params_}\n")
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means, stds, params):
			f.write(f"{mean} ({stdev}) with: {param}\n")

	# Display the best topology
	return grid_result.best_estimator_.model


if __name__ == "__main__":
	main()
