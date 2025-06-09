#!/usr/bin/env python
# Author: alejo.grillor02

# TODO: Write Docstrings...
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from sys import argv
from os import makedirs, path, environ
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import relu  # sigmoid, tanh
from tensorflow.keras.callbacks import ModelCheckpoint  # , EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import mae, logcosh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def main():

	def load_fold_data(fold_path, fold_number):
		"""Docstring..."""

		filename = f"{model_index}_{set_index}_fold_{fold_number}.txt"
		filepath = path.join(fold_path, filename)
		data = np.loadtxt(filepath)
		X = data[:, :-1]
		y = data[:, -1]
		return X, y

	def create_model(inputs: int, outputs: int):
		"""Docstring."""

		model = Sequential([
			Input(shape=(inputs,)),
			Dense(128, activation=relu, kernel_regularizer=l2(0.01)),
			BatchNormalization(),
			Dropout(0.5),
			Dense(64, activation=relu, kernel_regularizer=l2(0.01)),
			BatchNormalization(),
			Dropout(0.3),
			Dense(outputs)
		])

		return model

	model_index = argv[1]
	set_index = argv[2]
	val_fold = int(argv[3])

	config_path = path.join(environ['PROJECT_ROOT'], 'config.yaml')
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	N_FOLDS = config['N_FOLDS']
	EPOCHS = config['EPOCHS']
	BATCH_SIZE = config['BATCH_SIZE']
	LEARNING_RATE = config['LEARNING_RATE']

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
	X_train = np.delete(np.concatenate(X_train_parts, axis=0), 0, axis=1)
	y_train = np.concatenate(y_train_parts, axis=0)

	X_val = np.delete(X_val, 0, axis=1)

	# Create and compile model
	n_features = X_train.shape[1]
	n_labels = 1
	model = create_model(n_features, n_labels)
	model.compile(
		optimizer=Adam(learning_rate=LEARNING_RATE),
		loss=logcosh,
		metrics=[mae]
	)

	# Set up model checkpoint and other callbacks
	model_path = path.join(output_path, f'{model_index}_{set_index}_fold_{val_fold}.keras')
	checkpoint = ModelCheckpoint(
		model_path,
		save_best_only=True,
		monitor='val_loss',
		mode='auto',
		verbose=1
	)

	# early_stopping = EarlyStopping(
	# 	monitor='val_loss',
	# 	patience=100,  # Número de épocas sin mejora antes de parar
	# 	min_delta=0.0001,
	# 	restore_best_weights=True
	# )

	# reduce_lr = ReduceLROnPlateau(
	# 	monitor='val_loss',
	# 	factor=0.5,
	# 	patience=10,
	# 	min_lr=1e-8
	# )

	# Train the model
	stat = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		# callbacks=[early_stopping, reduce_lr, checkpoint],
		callbacks=[checkpoint],
		verbose=1
	)

	# Plot and save loss curve
	plt.figure(figsize=(8, 6))
	plt.plot(stat.history['loss'], label='Training Loss')
	plt.plot(stat.history['val_loss'], label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(path.join(output_path, f'{model_index}_{set_index}_fold_{val_fold}_loss.pdf'))
	plt.close()


if __name__ == "__main__":
	main()
