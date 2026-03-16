# dna-ssl-augmentations

Code for the paper:

**DNA augmentations for semi-supervised learning in genomic sequence classification**


This repository contains implementations of biologically motivated DNA sequence augmentations and the experimental framework used to evaluate them in semi-supervised learning (SSL) settings.

---

## Augmentations

Implemented DNA sequence augmentations include:

* reverse complement
* codon back-translation
* nucleotide substitution (NN)
* high-rate mutation
* length-preserving insertion/deletion (InDel)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dna-ssl-augmentations.git
cd dna-ssl-augmentations
pip install -r requirements.txt
```

---

## Example

Run an experiment:

```bash
python scripts/run_experiment.py
```

---


