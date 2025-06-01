#!/usr/bin/env python
# Author: alejo.grillor02

# TODO: Write Docstrings...
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from os import makedirs, path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # Dropout
from tensorflow.keras.activations import relu, sigmoid  # tanh, tanh
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import logcosh, mae
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
			Dense(16, activation=relu, input_shape=(inputs, )),
			Dense(8, activation=relu),
			Dense(outputs, activation=sigmoid)
		])

		return model

	training_set_path = argv[1]
	val_fold = int(argv[2])
	N_FOLDS = int(argv[3])
	EPOCHS = int(argv[4])
	BATCH_SIZE = int(argv[5])
	LEARNING_RATE = float(argv[6])

	# Parse the model and set index
	parts = training_set_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	output_path = argv[7]
	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	makedirs(output_path_base, exist_ok=True)

	# Cargar el set de training y de validación
	X_train_parts = []
	y_train_parts = []

	for fold_num in range(1, N_FOLDS + 1):
		if fold_num == val_fold:
			X_val, y_val = load_fold_data(training_set_path, fold_num)
		else:
			X, y = load_fold_data(training_set_path, fold_num)
			X_train_parts.append(X)
			y_train_parts.append(y)

	# Concatenate all training parts
	X_train = np.delete(np.concatenate(X_train_parts, axis=0), 0, axis=1)  # np.concatenate(X_train_parts, axis=0)
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

	# Set up model checkpoint to save the best model
	model_path = path.join(output_path_base, f'{model_index}_{set_index}_fold_{val_fold}.keras')
	checkpoint = ModelCheckpoint(
		model_path,
		save_best_only=True,
		monitor='val_loss',
		mode='auto',
		verbose=1
	)

	# Train the model
	stat = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
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
	plt.savefig(path.join(output_path_base, f'{model_index}_{set_index}_fold_{val_fold}_loss.pdf'))
	plt.close()

	# final_model_path = path.join(output_path_base, f'final_model_fold_{val_fold}.keras')
	# model.save(final_model_path)

	# Print final metrics
	# final_train_loss, final_train_acc = model.evaluate(X_train, y_train, verbose=0)
	# final_val_loss, final_val_acc = model.evaluate(X_val, y_val, verbose=0)


if __name__ == "__main__":
	main()
