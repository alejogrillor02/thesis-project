import numpy as np
import shap
from tensorflow.keras.models import load_model
import sys
from os import path


def main():
    # Parsear argumentos de línea de comandos
    model_path = sys.argv[1]
    training_set_path = sys.argv[2]
    val_fold = int(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) > 4 else "./"


if __name__ == "__main__":
    main()
