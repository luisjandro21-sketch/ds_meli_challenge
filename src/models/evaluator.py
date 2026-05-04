import numpy as np
from sklearn.metrics import classification_report

from src.config import ALERT_THRESHOLD, REFERENCE_METRICS

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


class ModelEvaluator:
    """Computes metrics and raises alerts when they deviate from the notebook baseline."""

    def __init__(self, threshold: float = ALERT_THRESHOLD):
        self.threshold = threshold

    def evaluate(self, y_true, y_pred) -> dict:
        report = classification_report(
            y_true, y_pred, target_names=["new", "used"], output_dict=True
        )
        return report

    def check_alerts(self, metrics: dict) -> list[str]:
        alerts = []

        for cls in ("new", "used"):
            ref_cls = REFERENCE_METRICS[cls]
            for metric in ("precision", "recall", "f1-score"):
                current = metrics[cls][metric]
                reference = ref_cls[metric]
                delta = current - reference
                if abs(delta) > self.threshold:
                    direction = "cayó" if delta < 0 else "subió"
                    alerts.append(
                        f"[ALERT] '{cls}' {metric} {direction} de "
                        f"{reference:.2f} a {current:.2f} (delta: {delta:+.3f})"
                    )

        # accuracy
        current_acc = metrics["accuracy"]
        ref_acc = REFERENCE_METRICS["accuracy"]
        delta_acc = current_acc - ref_acc
        if abs(delta_acc) > self.threshold:
            direction = "cayó" if delta_acc < 0 else "subió"
            alerts.append(
                f"[ALERT] accuracy {direction} de {ref_acc:.2f} a "
                f"{current_acc:.2f} (delta: {delta_acc:+.3f})"
            )

        return alerts

    def report(self, metrics: dict, alerts: list[str]) -> None:
        print(f"\n{_BOLD}=== Reporte de métricas ==={_RESET}")
        print(
            classification_report(
                [],
                [],
                target_names=["new", "used"],
                labels=[0, 1],
                zero_division=0,
            )
            if False  # placeholder; real print below
            else self._format_report(metrics)
        )

        if alerts:
            print(f"\n{_RED}{_BOLD}=== ALERTAS ({len(alerts)}) ==={_RESET}")
            for alert in alerts:
                print(f"  {_YELLOW}{alert}{_RESET}")
        else:
            print(f"\n{_GREEN}[OK] Todas las métricas dentro del rango esperado{_RESET}")

    # ------------------------------------------------------------------

    def _format_report(self, metrics: dict) -> str:
        lines = [
            f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
            "",
        ]
        for cls in ("new", "used"):
            m = metrics[cls]
            lines.append(
                f"{cls:>20} {m['precision']:>10.2f} {m['recall']:>10.2f} "
                f"{m['f1-score']:>10.2f} {int(m['support']):>10}"
            )
        lines += [
            "",
            f"{'accuracy':>20} {'':>10} {'':>10} {metrics['accuracy']:>10.2f} "
            f"{int(metrics['macro avg']['support']):>10}",
            f"{'macro avg':>20} {metrics['macro avg']['precision']:>10.2f} "
            f"{metrics['macro avg']['recall']:>10.2f} "
            f"{metrics['macro avg']['f1-score']:>10.2f} "
            f"{int(metrics['macro avg']['support']):>10}",
            f"{'weighted avg':>20} {metrics['weighted avg']['precision']:>10.2f} "
            f"{metrics['weighted avg']['recall']:>10.2f} "
            f"{metrics['weighted avg']['f1-score']:>10.2f} "
            f"{int(metrics['weighted avg']['support']):>10}",
        ]
        return "\n".join(lines)
