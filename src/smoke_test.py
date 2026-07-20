"""Minimal offline import check for the reproducible environment."""

import numpy
import pandas
import sklearn
import torch
import transformers

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"pandas={pandas.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print("Core imports: OK")
    print("AutoTokenizer import: OK")
    print("AutoModelForSequenceClassification import: OK")


if __name__ == "__main__":
    main()
