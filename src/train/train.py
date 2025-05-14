#!/usr/bin/env python
# Author: alejo.grillor02

# TODO: Write this Docstring.
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from os import makedirs, path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import relu  # tanh, sigmoid, tanh
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import logcosh, accuracy  # mae


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
			Dense(128, activation=relu, input_shape=(inputs, )),
			Dropout(0.3),
			Dense(64, activation=relu),
			Dropout(0.3),
			Dense(outputs, activation='softmax')
		])

		return model

	training_set_path = argv[1]
	val_fold = argv[2]
	N_FOLDS = argv[3]
	EPOCHS = argv[4]
	BATCH_SIZE = argv[5]

	# Parse the model and set index
	parts = training_set_path.strip("/").split("/")
	relevant_parts = parts[-2:]
	model_dir = relevant_parts[-2]  # 'model_XXX'
	set_dir = relevant_parts[-1]    # 'set_Y'
	model_index = model_dir.split("_")[1]
	set_index = set_dir.split("_")[1]

	output_path = argv[6] if len(argv) > 6 else "./"
	output_path_base = f"{output_path}/model_{model_index}/set_{set_index}"
	makedirs(output_path_base, exist_ok=True)

	# Cargar el set de training y de validación
	X_train_parts = []
	y_train_parts = []

	for fold_num in range(1, N_FOLDS):
		if fold_num == val_fold:
			X_val, y_val = load_fold_data(training_set_path, fold_num)
		else:
			X, y = load_fold_data(training_set_path, fold_num)
			X_train_parts.append(X)
			y_train_parts.append(y)

	# Concatenate all training parts
	X_train = np.concatenate(X_train_parts, axis=0)
	y_train = np.concatenate(y_train_parts, axis=0)

	# Create and compile model
	n_features = X_train.shape[1]
	n_labels = len(np.unique(y_train))
	model = create_model(n_features, n_labels)
	model.compile(
		optimizer='adam',
		loss=logcosh,
		metrics=[accuracy]
	)

	# Set up model checkpoint to save the best model
	model_path = path.join(output_path_base, f'{model_index}_{set_index}_fold_{val_fold}.h5')
	checkpoint = ModelCheckpoint(
		model_path,
		monitor='val_loss',
		save_best_only=True,
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
	plt.savefig(path.join(output_path, f'{model_index}_{set_index}_fold_{val_fold}_loss.pdf'))
	plt.close()

	# Plot and save accuracy curve
	plt.figure(figsize=(8, 6))
	plt.plot(stat.history['accuracy'], label='Training Accuracy')
	plt.plot(stat.history['val_accuracy'], label='Validation Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(path.join(output_path, f'{model_index}_{set_index}_fold_{val_fold}_accuracy.pdf'))
	plt.close()

	# final_model_path = path.join(output_path_base, f'final_model_fold_{val_fold}.h5')
	# model.save(final_model_path)

	# Print final metrics
	final_train_loss, final_train_acc = model.evaluate(X_train, y_train, verbose=0)
	final_val_loss, final_val_acc = model.evaluate(X_val, y_val, verbose=0)

	print(f"\nTraining completed for fold {val_fold}")
	print(f"Final Training Loss: {final_train_loss:.4f}, Accuracy: {final_train_acc:.4f}")
	print(f"Final Validation Loss: {final_val_loss:.4f}, Accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
	main()
