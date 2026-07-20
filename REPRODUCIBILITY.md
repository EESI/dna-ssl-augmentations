# Reproducible environment

This repository provides both a Conda environment and a Docker configuration.
Python and package versions are pinned to avoid dependency changes across installations.

## 1. Conda

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate dna-ssl-augmentations
python scripts/smoke_test.py
```

To remove and recreate the environment:

```bash
conda env remove -n dna-ssl-augmentations
conda env create -f environment.yml
```

## 2. Docker

Build the CPU-compatible image from the repository root:

```bash
docker build -t dna-ssl-augmentations .
```

Run the offline import smoke test:

```bash
docker run --rm dna-ssl-augmentations scripts/smoke_test.py
```

Show the training command help:

```bash
docker run --rm dna-ssl-augmentations src/train.py --help
```

Run training with local datasets mounted read-only and outputs mounted read-write:

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/outputs:/app/outputs" \
  dna-ssl-augmentations \
  src/train.py \
  --labeled_csv data/mytask/labeled.csv \
  --unlabeled_csv data/mytask/unlabeled.csv \
  --val_csv data/mytask/val.csv \
  --method fixmatch \
  --weak_aug nn \
  --strong_aug mutation \
  --output_dir outputs/my_fixmatch_run
```

The first model run downloads `PoetschLab/GROVER` from Hugging Face unless the model is already cached.
To preserve the cache between Docker runs, mount a cache directory:

```bash
docker run --rm \
  -v "$PWD/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/outputs:/app/outputs" \
  dna-ssl-augmentations \
  src/train.py --help
```

## 3. Verification

The following command verifies the package imports without downloading a model:

```bash
python scripts/smoke_test.py
```

A full model-loading check requires network access:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='PoetschLab/GROVER'; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m, num_labels=2); print('Model loading: OK')"
```
