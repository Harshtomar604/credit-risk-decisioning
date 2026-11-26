import json
from pathlib import Path

import joblib
import sklearn

from cv_compare import load_train
from pipelines import FEATURES, make_final_gbm, make_logreg

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_PATH = ROOT / "train_output.txt"


def main():
    X, y = load_train()
    MODEL_DIR.mkdir(exist_ok=True)

    models = {
        "gbm_isotonic": make_final_gbm(),
        "logistic_regression": make_logreg(),
    }

    lines = [f"training rows: {len(X)}, default rate: {y.mean():.4f}", ""]
    for name, model in models.items():
        model.fit(X, y)
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        p = model.predict_proba(X)[:, 1]
        lines.append(
            f"{name}: saved to {path.relative_to(ROOT).as_posix()} ({path.stat().st_size / 1024:.0f} KB), "
            f"mean predicted on train: {p.mean():.4f}"
        )

    meta = {
        "chosen_model": "gbm_isotonic",
        "benchmark_model": "logistic_regression",
        "features": FEATURES,
        "training_rows": int(len(X)),
        "training_default_rate": round(float(y.mean()), 4),
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    lines.append("metadata saved to models/metadata.json")

    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
