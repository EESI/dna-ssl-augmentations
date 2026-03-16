# dna-ssl-augmentations

Code for the paper:

**DNA augmentations for semi-supervised learning in genomic sequence classification**

This repository contains implementations of biologically motivated DNA sequence augmentations and the experimental framework used to evaluate them in semi-supervised learning (SSL) settings on genomic sequence classification tasks.

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
│   └── run_oncovirus_flexmatch.py
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



