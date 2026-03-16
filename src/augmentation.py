import random
import numpy as np

DNA_TO_AA_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

AA_TO_DNA_TABLE = {
    v: [k for k in DNA_TO_AA_TABLE if DNA_TO_AA_TABLE[k] == v]
    for v in set(DNA_TO_AA_TABLE.values())
}


def revcomp(seq: str) -> str:
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(complement)[::-1]


def dna_to_aa(sequence: str) -> str:
    return ''.join(DNA_TO_AA_TABLE.get(sequence[i:i+3], 'X') for i in range(0, len(sequence), 3))


def aa_to_dna(aa_sequence: str) -> str:
    return ''.join(np.random.choice(AA_TO_DNA_TABLE[aa]) for aa in aa_sequence if aa in AA_TO_DNA_TABLE)


def back_translation(sequence: str) -> str:
    aa_sequence = dna_to_aa(sequence)
    translated_dna = aa_to_dna(aa_sequence)
    return translated_dna


def weak_aug_bt(seq: str) -> str:
    s = revcomp(seq) if random.random() < 0.5 else seq
    return back_translation(s)


def strong_aug_bt(seq: str) -> str:
    s = revcomp(seq)
    return back_translation(s)


def weak_aug_nn(seq: str) -> str:
    mask_rate = 0.02
    mut_rate = 0.03

    s = revcomp(seq.upper()) if random.random() < 0.5 else seq.upper()
    s = list(s)
    bases = ['A', 'C', 'G', 'T']

    i = 0
    L = len(s)
    while i < L:
        r = random.random()
        if r < mask_rate:
            if i == L - 1:
                s[i] = 'N'
                i += 1
            else:
                s[i] = 'N'
                s[i + 1] = 'N'
                i += 2
            continue
        elif r < mask_rate + mut_rate and s[i] in bases:
            s[i] = random.choice([b for b in bases if b != s[i]])
        i += 1

    return ''.join(s)


def strong_aug_nn(seq: str) -> str:
    mask_rate = 0.02
    mut_rate = 0.03

    s = revcomp(seq.upper())
    s = list(s)
    bases = ['A', 'C', 'G', 'T']

    i = 0
    L = len(s)
    while i < L:
        r = random.random()
        if r < mask_rate:
            if i == L - 1:
                s[i] = 'N'
                i += 1
            else:
                s[i] = 'N'
                s[i + 1] = 'N'
                i += 2
            continue
        elif r < mask_rate + mut_rate and s[i] in bases:
            s[i] = random.choice([b for b in bases if b != s[i]])
        i += 1

    return ''.join(s)


def weak_aug_mutation(seq: str) -> str:
    mask_rate = 0.0
    mut_rate = 0.2

    s = revcomp(seq.upper()) if random.random() < 0.5 else seq.upper()
    s = list(s)
    bases = ['A', 'C', 'G', 'T']

    i = 0
    L = len(s)
    while i < L:
        r = random.random()
        if r < mask_rate:
            if i == L - 1:
                s[i] = 'N'
                i += 1
            else:
                s[i] = 'N'
                s[i + 1] = 'N'
                i += 2
            continue
        elif r < mask_rate + mut_rate and s[i] in bases:
            s[i] = random.choice([b for b in bases if b != s[i]])
        i += 1

    return ''.join(s)


def strong_aug_mutation(seq: str) -> str:
    mask_rate = 0.0
    mut_rate = 0.2

    s = revcomp(seq.upper())
    s = list(s)
    bases = ['A', 'C', 'G', 'T']

    i = 0
    L = len(s)
    while i < L:
        r = random.random()
        if r < mask_rate:
            if i == L - 1:
                s[i] = 'N'
                i += 1
            else:
                s[i] = 'N'
                s[i + 1] = 'N'
                i += 2
            continue
        elif r < mask_rate + mut_rate and s[i] in bases:
            s[i] = random.choice([b for b in bases if b != s[i]])
        i += 1

    return ''.join(s)


def weak_aug_indel(seq: str) -> str:
    ratio = 0.05
    s = revcomp(seq.upper()) if random.random() < 0.5 else seq.upper()
    s = list(s)
    bases = ['A', 'C', 'G', 'T']
    L = len(s)

    n_mut = max(1, int(L * ratio))
    del_positions = sorted(random.sample(range(L), n_mut))
    for idx in reversed(del_positions):
        del s[idx]

    new_L = len(s)
    ins_positions = sorted(random.sample(range(new_L + 1), n_mut))
    for offset, idx in enumerate(ins_positions):
        s.insert(idx + offset, random.choice(bases))

    assert len(s) == L, f"Length changed: {len(s)} != {L}"
    return ''.join(s)


def strong_aug_indel(seq: str, ratio: float = 0.05) -> str:
    bases = ['A', 'C', 'G', 'T']
    s = list(revcomp(seq.upper()))
    L = len(s)

    n_mut = max(1, int(L * ratio))
    del_positions = sorted(random.sample(range(L), n_mut))
    for idx in reversed(del_positions):
        del s[idx]

    new_L = len(s)
    ins_positions = sorted(random.sample(range(new_L + 1), n_mut))
    for offset, idx in enumerate(ins_positions):
        s.insert(idx + offset, random.choice(bases))

    assert len(s) == L, f"Length changed: {len(s)} != {L}"
    return ''.join(s)


def weak_aug_indelnn(seq: str, ratio: float = 0.05) -> str:
    mask_rate = 0.02
    mut_rate = 0.03
    bases = ['A', 'C', 'G', 'T']

    s = revcomp(seq.upper()) if random.random() < 0.5 else seq.upper()
    s = list(s)
    i = 0
    L = len(s)

    while i < L:
        r = random.random()
        if r < mask_rate:
            if i == L - 1:
                s[i] = 'N'
                i += 1
            else:
                s[i] = 'N'
                s[i + 1] = 'N'
                i += 2
            continue
        elif r < mask_rate + mut_rate and s[i] in bases:
            s[i] = random.choice([b for b in bases if b != s[i]])
        i += 1

    n_mut = max(1, int(L * ratio))
    del_positions = sorted(random.sample(range(L), n_mut))
    for idx in reversed(del_positions):
        del s[idx]

    new_L = len(s)
    ins_positions = sorted(random.sample(range(new_L + 1), n_mut))
    for offset, idx in enumerate(ins_positions):
        s.insert(idx + offset, random.choice(bases))

    assert len(s) == L, f"Length changed: {len(s)} != {L}"
    return ''.join(s)


def strong_aug_indelnn(seq: str, ratio: float = 0.05) -> str:
    mask_rate = 0.02
    mut_rate = 0.03
    bases = ['A', 'C', 'G', 'T']

    s = list(revcomp(seq.upper()))
    i = 0
    L = len(s)

    while i < L:
        r = random.random()
        if r < mask_rate:
            if i == L - 1:
                s[i] = 'N'
                i += 1
            else:
                s[i] = 'N'
                s[i + 1] = 'N'
                i += 2
            continue
        elif r < mask_rate + mut_rate and s[i] in bases:
            s[i] = random.choice([b for b in bases if b != s[i]])
        i += 1

    n_mut = max(1, int(L * ratio))
    del_positions = sorted(random.sample(range(L), n_mut))
    for idx in reversed(del_positions):
        del s[idx]

    new_L = len(s)
    ins_positions = sorted(random.sample(range(new_L + 1), n_mut))
    for offset, idx in enumerate(ins_positions):
        s.insert(idx + offset, random.choice(bases))

    assert len(s) == L, f"Length changed: {len(s)} != {L}"
    return ''.join(s)


AUGMENT_FUNCS = {
    "bt": (weak_aug_bt, strong_aug_bt),
    "nn": (weak_aug_nn, strong_aug_nn),
    "mutation": (weak_aug_mutation, strong_aug_mutation),
    "indel": (weak_aug_indel, strong_aug_indel),
    "indelnn": (weak_aug_indelnn, strong_aug_indelnn),
}


def build_augment_pairs():
    pairs = []
    for weak_name, (weak_fn, _) in AUGMENT_FUNCS.items():
        for strong_name, (_, strong_fn) in AUGMENT_FUNCS.items():
            pairs.append({
                "weak_name": weak_name,
                "strong_name": strong_name,
                "weak_fn": weak_fn,
                "strong_fn": strong_fn,
            })
    return pairs
