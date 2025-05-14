#!/usr/bin/env python

"""
PUBLIC DOCSTRING.

PLACEHOLDER
"""

import numpy as np
import matplotlib.pylab as pl
from sys import argv
from os import path


def comp() -> None:
	"""Docstring."""
	input_model = argv[1]
	input_test = argv[2]

	in_basename = path.basename(input_model)
	in_lens_model = in_basename[:3]
	in_set_index = in_basename[4]
	in_sample_index = in_basename[6]

	test_basename = path.basename(input_test)
	test_lens_model = test_basename[:3]
	test_set_index = test_basename[4]
	test_sample_index = test_basename[6]

	output_dir = "./output/" + in_lens_model + "/" + in_set_index + "/test/" + in_sample_index
	commonname = in_lens_model + "-" + in_set_index + "-" + in_sample_index + "->" + test_lens_model + "-" + test_set_index + "-" + test_sample_index
	output_fig_path = output_dir + "/error-" + commonname + "-comp.png"

	data = np.loadtxt(output_dir + "/error-" + commonname + ".txt")
	error = np.abs(data[:, -1])
	count = len(data[:, -1])

	error = error[error >= 0.5]
	count_mal = len(error)
	error1 = error[error >= 1.0]
	count_mal1 = len(error1)

	porc = ((count_mal / count) * 100)
	porc1 = ((count_mal1 / count) * 100)

	val_const = [0.5] * len(data)

	pl.figure(1)
	x = np.arange(count)
	pl.scatter(x, data[:, -1], label="error")
	pl.plot(x, np.array(val_const), color="r")
	pl.plot(x, np.array(val_const) * (-1), color="r")
	pl.title("Ploteo de los errores: " + str(round(porc, 2)) + " " + str(round(porc1, 2)))
	pl.legend()
	pl.savefig(output_fig_path, dpi=300)
	# pl.show()


if __name__ == "__main__":
	comp()
