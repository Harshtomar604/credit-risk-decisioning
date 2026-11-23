import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import build_features  # noqa: E402
import calibration  # noqa: E402
import cv_compare  # noqa: E402
import explain_fairness  # noqa: E402
import load_data  # noqa: E402
import profit_threshold  # noqa: E402
import split_data  # noqa: E402
import train_final  # noqa: E402

STEPS = [
    ("1. load data into DuckDB", load_data),
    ("2. build SQL features", build_features),
    ("3. stratified train/test split", split_data),
    ("4. cross-validated model comparison", cv_compare),
    ("5. calibration check", calibration),
    ("6. train and save final models", train_final),
    ("7. profit threshold on test set", profit_threshold),
    ("8. permutation importance and fairness audit", explain_fairness),
]


def main():
    for name, module in STEPS:
        print(f"\n===== {name} =====")
        module.main()
    print("\nPipeline complete. Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
