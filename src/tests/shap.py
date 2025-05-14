import numpy as np
import shap
from tensorflow.keras.models import load_model
import sys
from os import path


def load_fold_data(fold_path, fold_number, model_index, set_index):
    """Carga los datos de un fold específico."""
    filename = f"{model_index}_{set_index}_fold_{fold_number}.txt"
    filepath = path.join(fold_path, filename)
    data = np.loadtxt(filepath)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def main():
    # Parsear argumentos de línea de comandos
    if len(sys.argv) < 4:
        print("Uso: python shap_script.py <ruta_modelo.h5> <ruta_datos_entrenamiento> <val_fold> [ruta_salida]")
        sys.exit(1)

    model_path = sys.argv[1]
    training_set_path = sys.argv[2]
    val_fold = int(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) > 4 else "./"

    # Extraer model_index y set_index del nombre del modelo
    model_filename = path.basename(model_path)
    parts = model_filename.split('_')
    model_index = parts[0]
    set_index = parts[1]

    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Cargar datos de validación
    X_val, y_val = load_fold_data(training_set_path, val_fold, model_index, set_index)

    # Usar subconjunto para cálculo eficiente
    background = X_val[:100]
    samples = X_val[:100]

    # Inicializar explainer de SHAP
    explainer = shap.DeepExplainer(model, background)

    # Calcular valores SHAP
    shap_values = explainer.shap_values(samples)

    # Calcular importancia de features
    if isinstance(shap_values, list):
        # Para múltiples clases: promedio absoluto a través de clases y muestras
        mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
        feature_importance = np.mean(mean_abs_per_class, axis=0)
    else:
        # Clasificación binaria
        feature_importance = np.mean(np.abs(shap_values), axis=0)

    # Guardar resultados
    output_file = path.join(output_path, f'shap_importance_{model_index}_{set_index}_fold_{val_fold}.txt')
    np.savetxt(output_file, feature_importance)

    print(f"Importancia de features guardada en: {output_file}")


if __name__ == "__main__":
    main()
