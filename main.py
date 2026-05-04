import argparse

from src.config import DATA_PATH
from src.pipeline import MLPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificador new/used para listings de MercadoLibre."
    )
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help="Ruta al archivo .jsonlines (default: MLA_100k.jsonlines en el proyecto).",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        metavar="PATH",
        help="Ruta donde guardar el modelo entrenado (joblib). Ej: model.pkl",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Umbral de alerta en puntos porcentuales (default: 0.03).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = MLPipeline(alert_threshold=args.alert_threshold)
    pipeline.run(data_path=args.data_path, save_model_path=args.save_model)


if __name__ == "__main__":
    main()
