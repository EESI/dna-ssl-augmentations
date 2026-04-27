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
│   ├── fasta_to_csv.py
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


---

### Semi-Supervised Fine-Tuned Models

Semi-supervised fine-tuned models for AMR and Oncovirus classification are available on Zenodo:

https://doi.org/10.5281/zenodo.19671648

Each model archive contains the files required for inference. After downloading and extracting, use the extracted directory as `--model_dir`.

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

### Dataset Access

The AMR datasets used in this project can be obtained from the following sources:

https://drive.google.com/drive/folders/1GSVMmW-T3E0ua94qxzU-lXU3-Ozxp7op?usp=sharing



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




## FASTA support

This repository expects CSV inputs for training and inference.
If your data is in FASTA format, you can convert it to the required CSV format using the provided utility script as shown below:




### 1) unlabeled FASTA → CSV

```bash
python src/fasta_to_csv.py \
  --input_fasta data/mytask/unlabeled.fasta \
  --output_csv data/mytask/unlabeled.csv \
  --unlabeled
```

Output:

```csv
sequence
ACGTACGTACGT
TTGCAATGCCAA
```

### 2) labeled FASTA, header as a label

FASTA:

```text
>class0
ACGTACGTACGT
>class1
TTGCAATGCCAA
```

Command:

```bash
python src/fasta_to_csv.py \
  --input_fasta data/mytask/labeled.fasta \
  --output_csv data/mytask/labeled.csv \
  --label-from-header
```

Output:

```csv
sequence,label
ACGTACGTACGT,class0
TTGCAATGCCAA,class1
```

### 3) labeled FASTA, parse header with delimiter to extract label

FASTA:

```text
>seq001|0
ACGTACGTACGT
>seq002|1
TTGCAATGCCAA
```

Command:

```bash
python src/fasta_to_csv.py \
  --input_fasta data/mytask/labeled.fasta \
  --output_csv data/mytask/labeled.csv \
  --header-split-delim "|" \
  --label-index 1
```

### 4) labeled FASTA, key=value format to extract label 

FASTA:

```text
>seq001 sample=a label=0
ACGTACGTACGT
>seq002 sample=b label=1
TTGCAATGCCAA
```

Command:

```bash
python src/fasta_to_csv.py \
  --input_fasta data/mytask/labeled.fasta \
  --output_csv data/mytask/labeled.csv \
  --label-key label
```


### Training with FASTA(convert to CSV first)

```bash
python src/fasta_to_csv.py \
  --input_fasta data/mytask/labeled.fasta \
  --output_csv data/mytask/labeled.csv \
  --header-split-delim "|" \
  --label-index 1

python src/fasta_to_csv.py \
  --input_fasta data/mytask/unlabeled.fasta \
  --output_csv data/mytask/unlabeled.csv \
  --unlabeled

python src/fasta_to_csv.py \
  --input_fasta data/mytask/val.fasta \
  --output_csv data/mytask/val.csv \
  --header-split-delim "|" \
  --label-index 1

python src/fasta_to_csv.py \
  --input_fasta data/mytask/test.fasta \
  --output_csv data/mytask/test.csv \
  --header-split-delim "|" \
  --label-index 1
```

FixMatch:

```bash
python src/train.py \
  --labeled_csv data/mytask/labeled.csv \
  --unlabeled_csv data/mytask/unlabeled.csv \
  --val_csv data/mytask/val.csv \
  --test_csv data/mytask/test.csv \
  --method fixmatch \
  --weak_aug nn \
  --strong_aug mutation \
  --output_dir outputs/my_fixmatch_run
```

FlexMatch:

```bash
python src/train.py \
  --labeled_csv data/mytask/labeled.csv \
  --unlabeled_csv data/mytask/unlabeled.csv \
  --val_csv data/mytask/val.csv \
  --test_csv data/mytask/test.csv \
  --method flexmatch \
  --weak_aug nn \
  --strong_aug mutation \
  --output_dir outputs/my_flexmatch_run
```

### Inference with FASTA(convert FASTA to CSV fisrt)

```bash
python src/fasta_to_csv.py \
  --input_fasta data/mytask/new_sequences.fasta \
  --output_csv data/mytask/new_sequences.csv \
  --unlabeled
```

inference:

```bash
python src/inference.py \
  --model_dir outputs/my_flexmatch_run \
  --input_csv data/mytask/new_sequences.csv \
  --output_csv outputs/my_flexmatch_run/predictions.csv
```





## Auto FASTA Pipeline

The Auto FASTA Pipeline allows you to use FASTA files directly for training and inference without manually converting them to CSV.  
It automatically converts FASTA → CSV internally and runs the existing pipeline.

---

### Label format (required)

For labeled FASTA, the label must be an **integer placed after a space at the end of the header**:

```fasta
>seq001 0
ACGTACGTACGT
>seq002 1
TTGCAATGCCAA
````

* The last whitespace-separated token is used as the label
* Labels must be integers (0, 1, 2, ...)

Unlabeled FASTA:

```fasta
>seq001
ACGTACGTACGT
```

---

### Train

```bash
python src/fasta_auto_pipeline.py \
  --mode train \
  --labeled_fasta data/labeled.fasta \
  --unlabeled_fasta data/unlabeled.fasta \
  --val_fasta data/val.fasta \
  --test_fasta data/test.fasta \
  --method fixmatch \
  --weak_aug nn \
  --strong_aug mutation \
  --output_dir outputs/run
```

---

### Inference

```bash
python src/fasta_auto_pipeline.py \
  --mode inference \
  --input_fasta data/test.fasta \
  --model_dir outputs/run \
  --output_csv outputs/run/predictions.csv
```



