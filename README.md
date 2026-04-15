# dna-ssl-augmentations

Code for the paper:

**DNA augmentations for semi-supervised learning in genomic sequence classification**

This repository contains implementations of biologically motivated DNA sequence augmentations and the experimental framework used to evaluate them in semi-supervised learning (SSL) settings on genomic sequence classification tasks.

This repository now provides a user-oriented training and inference pipeline for applying DNA augmentation-based SSL methods to custom genomic datasets.


## Features

This repository currently supports:

- biologically motivated DNA sequence augmentations
- semi-supervised learning with **FixMatch**
- semi-supervised learning with **FlexMatch**
- experiments on:
  - **AMR**
  - **Oncovirus**

## Implemented augmentations

Implemented DNA sequence augmentations include:

- reverse complement
- codon back-translation
- nucleotide masking/substitution (`NN`)
- high-rate mutation
- length-preserving insertion/deletion (`InDel`)
- combined `InDel + NN`

## Repository structure

```text
dna-ssl-augmentations/
├── src/
│   ├── augmentations.py
│   ├── common.py
│   ├── fixmatch_core.py
│   ├── flexmatch_core.py
│   ├── run_amr_fixmatch.py
│   ├── run_amr_flexmatch.py
│   ├── run_oncovirus_fixmatch.py
│   ├── run_oncovirus_flexmatch.py
│   ├── train.py
│   └── inference.py
├── data/
│   ├── amr/
│   └── oncovirus/
├── outputs/
├── requirements.txt
└── README.md


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/EESI/dna-ssl-augmentations.git
cd dna-ssl-augmentations
pip install -r requirements.txt
````
---
## Quick Start (Using Your Own Data)

This package supports training semi-supervised genomic sequence classifiers using your own labeled and unlabeled datasets.

### Input format

#### Labeled / validation / test CSV

```csv
sequence,label
ACGTACGTACGT,0
TTGCAATGCCAA,1
````

#### Unlabeled 
```csv
sequence
ACGTACGTACGT
TTGCAATGCCAA
````


## Training

### FixMatch

```bash
python src/train.py \
  --labeled_csv data/mytask/labeled.csv \
  --unlabeled_csv data/mytask/unlabeled.csv \
  --val_csv data/mytask/val.csv \
  --test_csv data/mytask/test.csv \
  --method fixmatch \
  --weak_aug nn \
  --strong_aug mutation \
  --model_name PoetschLab/GROVER \
  --output_dir outputs/my_fixmatch_run
```

### FlexMatch

```bash
python src/train.py \
  --labeled_csv data/mytask/labeled.csv \
  --unlabeled_csv data/mytask/unlabeled.csv \
  --val_csv data/mytask/val.csv \
  --method flexmatch \
  --weak_aug nn \
  --strong_aug mutation \
  --model_name PoetschLab/GROVER \
  --output_dir outputs/my_flexmatch_run
```

* `--test_csv` is optional
* `--weak_aug` and `--strong_aug` can be selected from:
  `bt`, `nn`, `mutation`, `indel`, `indelnn`

After training, the following files will be saved in `output_dir`:

* `model_state.pt`
* `label_map.json`
* `train_config.json`
* tokenizer files

---

## Inference

Run prediction on new sequences:

```bash
python src/inference.py \
  --model_dir outputs/my_flexmatch_run \
  --input_csv data/mytask/new_sequences.csv \
  --output_csv outputs/my_flexmatch_run/predictions.csv
```

If your sequence column is not named `sequence`:

```bash
python src/inference.py \
  --model_dir outputs/my_flexmatch_run \
  --input_csv data/mytask/new_sequences.csv \
  --sequence_col "DNA Sequence" \
  --output_csv outputs/my_flexmatch_run/predictions.csv
```

---

### Inference input format

```csv
sequence
ACGTACGTACGT
TTGCAATGCCAA
```

---

### Output format

```csv
sequence,pred_id,pred_label,pred_confidence,prob_0,prob_1
ACGTACGTACGT,1,1,0.9321,0.0679,0.9321
TTGCAATGCCAA,0,0,0.8812,0.8812,0.1188
```

````











---

## Datasets

This repository does **not** redistribute the datasets.

Please download or prepare the datasets yourself and place them in the following locations.

### AMR

Expected files:

```text
data/amr/train_6classes.csv
data/amr/val_6classes.csv
data/amr/test_6classes.csv
```

### Oncovirus

Expected files:

```text
data/oncovirus/train.csv
data/oncovirus/val.csv
data/oncovirus/test.csv
```

---

## Expected CSV format

Each CSV file should contain at least the following columns:

```text
sequence,label
```

If your sequence column is named `DNA Sequence`, it will be automatically renamed internally.

---

## Running experiments

### AMR + FixMatch

```bash
python src/run_amr_fixmatch.py
```

### AMR + FlexMatch

```bash
python src/run_amr_flexmatch.py
```

### Oncovirus + FixMatch

```bash
python src/run_oncovirus_fixmatch.py
```

### Oncovirus + FlexMatch

```bash
python src/run_oncovirus_flexmatch.py
```

```



