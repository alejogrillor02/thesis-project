#!/usr/bin/env python

# TODO: Write Docstrings... and fix missing vars
"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

from sys import argv
from os import makedirs, path
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid  # tanh, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import logcosh, mae
from tensorflow.keras.optimizers import Adam


def main():

	def create_model(layers=2, units=16, dropout_rate=0.0, l2_reg=0.0, learning_rate=0.001):
		model = Sequential()

		for i in range(layers):
			model.add(Dense(units, activation=relu, kernel_regularizer=l2(l2_reg)))
			if dropout_rate > 0:
				model.add(Dropout(dropout_rate))

		model.add(Dense(1, activation=sigmoid))

		model.compile(
			optimizer=Adam(learning_rate=learning_rate),
			loss=logcosh,
			metrics=[mae]
		)
		return model

	model = KerasRegressor(build_fn=create_model, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

	param_grid = {
		'layers': [1, 2, 3],
		'units': [8, 16, 32, 64],
		'dropout_rate': [0.0, 0.2, 0.4],
		'l2_reg': [0.0, 1e-4, 1e-3],
		'learning_rate': [0.001, 0.0001]
	}

	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, scoring='neg_mean_absolute_error')
	grid_result = grid.fit(X_train, y_train)

	print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
	return grid_result.best_estimator_.model


if __name__ == "__main__":
	main()
