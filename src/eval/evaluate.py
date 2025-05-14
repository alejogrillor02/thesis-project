#!/usr/bin/env python
#
# License: GNU GPLv3
# python 3.11.5
# tensorflow 2.13.0

"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from sys import argv
from os import path, makedirs

# from tensorflow.keras.losses import mae
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
# from tensorflow.keras.activations import sigmoid
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


def evaluate_with_keras() -> None:
	"""Docstring."""
	input_model = argv[1]
	input_test = argv[2]

	def denormalizeminmax(array: np.array, cmax: float, cmin: float):
		"""Docstring."""
		return np.array([normalized_value * (cmax - cmin) + cmin for normalized_value in array])

	def returnminmax(lens_model: str, set_index: str):
		"""Docstring."""
		with open("data/processed/" + lens_model + "/" + lens_model + "-" + set_index + "-Normalization-data.csv", "r") as file:
			minmax = [row[0:2] for idx, row in enumerate(csv.reader(file)) if idx == 5][0]
		return [float(x) for x in minmax]

	in_basename = path.basename(input_model)
	in_lens_model = in_basename[:3]
	in_set_index = in_basename[4]
	in_sample_index = in_basename[6]

	test_basename = path.basename(input_test)
	test_lens_model = test_basename[:3]
	test_set_index = test_basename[4]
	test_sample_index = test_basename[6]

	output_dir = "./output/" + in_lens_model + "/" + in_set_index + "/test/" + in_sample_index
	makedirs(output_dir, exist_ok=True)
	commonname = in_lens_model + "-" + in_set_index + "-" + in_sample_index + "->" + test_lens_model + "-" + test_set_index + "-" + test_sample_index
	output_fig_path = output_dir + "/error-" + commonname + "-test.png"
	output_error_path = output_dir + "/error-" + commonname + ".txt"
	output_mean_path = output_dir + "/error-" + commonname + "-mean.txt"

	model = load_model(input_model)
	# model.summary()

	DATA_IN = [0, 1, 2, 3, 4]
	DATA_OUT = [5]

	# TODO: Revisar porque rayos creé test_orig, prefiero no tocarlo
	test_orig = np.loadtxt(input_test)
	test = np.loadtxt(input_test)

	test_data = test[:, DATA_IN], test[:, DATA_OUT]

	predicted_y = model.predict(test_data[0])
	real = test_data[1]

	in_cmax, in_cmin = returnminmax(in_lens_model, in_set_index)
	test_cmax, test_cmin = returnminmax(in_lens_model, in_set_index)
	denorm_predicted_y = denormalizeminmax(predicted_y, in_cmax, in_cmin)
	denorm_real = denormalizeminmax(real, test_cmax, test_cmin)

	error = (denorm_predicted_y - denorm_real).flatten()
	error_mean = error.mean()
	file = open(output_mean_path, "w")
	file.write("%.2f" % (error_mean))
	plot_data = np.empty((len(real), 8))

	plot_data[:, 0] = test_orig[:, 0]
	plot_data[:, 1] = test_orig[:, 1]
	plot_data[:, 2] = test_orig[:, 2]
	plot_data[:, 3] = test_orig[:, 3]
	plot_data[:, 4] = test_orig[:, 4]
	plot_data[:, 5] = test_orig[:, 5]
	plot_data[:, 6] = denorm_predicted_y.flatten()
	plot_data[:, 7] = error

	np.savetxt(output_error_path, plot_data, fmt="%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f")

	# sns.set(font_scale=1)
	plt.subplot(2, 1, 1)
	plt.figure(1)
	plt.title("Potencia óptica")

	x = np.arange(0, len(denorm_predicted_y))
	plt.scatter(x, denorm_real, label="Real values")
	plt.scatter(x, denorm_predicted_y, label="Predicted")
	plt.legend()
	plt.subplot(2, 1, 2)
	plt.plot(error)
	plt.ylabel("Error")
	plt.savefig(output_fig_path, dpi=300)
	# plt.show()


if __name__ == "__main__":
	evaluate_with_keras()
