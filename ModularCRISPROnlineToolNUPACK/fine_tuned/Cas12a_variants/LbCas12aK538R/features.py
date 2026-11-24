# features.py
import numpy as np
import torch
from torch_geometric.data import Data

from nupack import Model, pairs
from nupack import *

TEST_GUIDE_LEN = 20
TEST_TARGET_LEN = 20
scaffold = "UAAUUUCUACUAAGUGUAGAU"

# Define NUPACK model
my_model_RNA = Model(material='rna', ensemble='nostacking', celsius=37, sodium=0.1, magnesium=0.01)
my_model_DNA = Model(material='dna', ensemble='nostacking', celsius=37, sodium=0.1, magnesium=0.01)

def clean_seq(seq: str) -> str:
    return seq.strip().upper().replace(" ", "").replace("\n", "")

def validate_seq(seq: str, seq_type: str):
    allowed = set("ACGTU")

    if seq_type == "guide":
        expected_len = TEST_GUIDE_LEN
    elif seq_type == "target":
        expected_len = TEST_TARGET_LEN
    else:
        raise ValueError(f"Unknown seq_type: {seq_type}")

    if len(seq) != expected_len:
        raise ValueError(f"{seq_type} must be exactly {expected_len} nt.")

    bad = [c for c in seq if c not in allowed]
    if bad:
        raise ValueError(f"Invalid bases in {seq_type}: {sorted(set(bad))}")
    
def nucleotide_to_one_hot(nucleotide):
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'U': [0, 0, 0, 1],
        '-': [0, 0, 0, 0]
    }
    return mapping.get(nucleotide, [0, 0, 0, 0])

def encode_sequence(sequence):
    return [nucleotide_to_one_hot(nuc) for nuc in sequence]

def pad_sequence(seq_encoded, max_len=30):
    padding_len = max_len - len(seq_encoded)
    if padding_len > 0:
        seq_encoded = seq_encoded + [[0, 0, 0, 0]] * padding_len
    else:
        seq_encoded = seq_encoded[:max_len]  # truncate if longer
    return seq_encoded

# -----------------------------
# 1. Sequence features: x1
# shape should be (1, 60, 4)
# -----------------------------
def build_sequence_features(guide, target):
    target = target[::-1]  # Since Cas12a PAM-proximal end is located at the 5' of spacer

    guide_encoded = encode_sequence(guide)
    target_encoded = encode_sequence(target)

    guide_padded = pad_sequence(guide_encoded, 30)
    target_padded = pad_sequence(target_encoded, 30)

    seq_array_unit = guide_padded + target_padded 

    arr = np.array(seq_array_unit, dtype=np.float32).reshape(60, 4)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0) 

# -----------------------------
# 2. Structure features: x2
# shape should be (1, 90, 3)
# -----------------------------
def DNA_reverse_complement(DNA):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(DNA))

def get_segments_interrupted_by_plus(input_string: str) -> list:
    # Split the string by the "+" character
    segments = input_string.split("+")
    
    # Remove empty segments if any (e.g., if the string starts or ends with "+")
    segments = [segment for segment in segments if segment]
    
    return segments

def build_structure_features(guide, target):
    struct_array_unit = []

    guide = scaffold + guide

    # Compute suboptimal structures and energy
    subopt_structures_guide_bh = subopt(strands=guide, energy_gap=0.1, model=my_model_RNA)  
    subopt_structures_target_ssDNA = subopt(strands=target, energy_gap=0.1, model=my_model_DNA)
    subopt_structures_hybrid_ah = subopt(strands=[guide, target], energy_gap=0.1, model=my_model_DNA)

    # Guide structure before hybridization
    spacer_struct_bh = str(subopt_structures_guide_bh[0].structure)[len(scaffold):len(scaffold)+TEST_TARGET_LEN]
    for j in range (0, 30):
        
        try:
            if spacer_struct_bh[j] == '.':
                to_be_added = ([1, 0, 0])
            elif spacer_struct_bh[j] == '(':
                to_be_added = ([0, 1, 0])
            elif spacer_struct_bh[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
                
        except IndexError:
            to_be_added = ([0, 0, 0])

        struct_array_unit.append(to_be_added)
        
    # ssDNA target structure before hybridization 
    target_ssDNA_struct_bh = (str(subopt_structures_target_ssDNA[0].structure))

    for j in range (0, 30):
        
        try:
            if target_ssDNA_struct_bh[j] == '.':
                to_be_added = ([1, 0, 0])
            elif target_ssDNA_struct_bh[j] == '(':
                to_be_added = ([0, 1, 0])
            elif target_ssDNA_struct_bh[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)

    # Hybrid structure after hybridization 
    hybrid_ah_1 = get_segments_interrupted_by_plus(str(subopt_structures_hybrid_ah[0].structure))[0]
    spacer_struct_ah = hybrid_ah_1[len(scaffold):len(scaffold)+TEST_TARGET_LEN]

    for j in range (0, 30):
        
        try:
            if spacer_struct_ah[j] == '.':
                to_be_added = ([1, 0, 0])
            elif spacer_struct_ah[j] == '(':
                to_be_added = ([0, 1, 0])
            elif spacer_struct_ah[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)

    arr = np.array(struct_array_unit, dtype=np.float32).reshape(90, 3)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# 3. Energy features: x3
# shape should be (1, 120)
# -----------------------------
def RNA_to_DNA(RNA):
    match = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'T'}
    return ''.join(match.get(base, base) for base in (RNA))

def RNA_reverse_complement(RNA):
    complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(RNA))

def normalize_guide(value):
    normalized_value = (value - (-15.93)) / (0 - (-15.93))
    return normalized_value

def normalize_target(value):
    normalized_value = (value - (-55.53)) / (-11.88 - (-55.53))
    return normalized_value

def normalize_ssDNA_target_bh(value):
    normalized_value = (value - (-12.20)) / (0 - (-12.20))
    return normalized_value

def normalize_guide_conse(value):
    normalized_value = (value - (-52.06)) / (2.61 - (-52.06))
    return normalized_value

def normalize_target_conse(value):
    normalized_value = (value - (-52.06)) / (0 - (-52.06))
    return normalized_value

def normalize_ssDNA_target_bh_conse(value):
    normalized_value = (value - (-15.21)) / (0 - (-15.21))
    return normalized_value

def normalize_guide_conse_unpaired(value):
    normalized_value = (value - (-52.06)) / (2.61 - (-52.06))
    return normalized_value

def normalize_target_conse_unpaired(value):
    normalized_value = (value - (-52.06)) / (0 - (-52.06))
    return normalized_value

def normalize_ssDNA_target_bh_conse_unpaired(value):
    normalized_value = (value - (-33.55)) / (0 - (-33.55))
    return normalized_value

def normalize_guide_overhang(value):
    normalized_value = (value - (-52.06)) / (2.61 - (-52.06))
    return normalized_value

def normalize_target_overhang(value):
    normalized_value = (value - (-52.06)) / (0 - (-52.06))
    return normalized_value

def normalize_ssDNA_target_bh_overhang(value):
    normalized_value = (value - (-33.55)) / (0 - (-33.55))
    return normalized_value

def normalize_guide_paired(value):
    normalized_value = (value - (-52.06)) / (2.61 - (-52.06))
    return normalized_value

def normalize_target_paired(value):
    normalized_value = (value - (-52.06)) / (0 - (-52.06))
    return normalized_value

def normalize_ssDNA_target_bh_paired(value):
    normalized_value = (value - (-15.21)) / (0 - (-15.21))
    return normalized_value

def normalize_seed(value):
    normalized_value = (value - (-10.99)) / (-3.33 - (-10.99))
    return normalized_value

def normalize_middle(value):
    normalized_value = (value - (-13.07)) / (-4.12 - (-13.07))
    return normalized_value

def normalize_distal(value):
    normalized_value = (value - (-13.07)) / (-4.12 - (-13.07))
    return normalized_value

def normalize_target_conse_3(value):
    normalized_value = (value - (-3.89)) / (0.41 - (-3.89))
    return normalized_value

def normalize_target_conse_4(value):
    normalized_value = (value - (-7.20)) / (-0.80 - (-7.20))
    return normalized_value

def normalize_target_conse_5(value):
    normalized_value = (value - (-9.50)) / (-1.80 - (-9.50))
    return normalized_value

def normalize_target_conse_6(value):
    normalized_value = (value - (-12.80)) / (-3.00 - (-12.80))
    return normalized_value

def normalize_target_conse_7(value):
    normalized_value = (value - (-15.11)) / (-4.01 - (-15.11))
    return normalized_value

def normalize_target_conse_8(value):
    normalized_value = (value - (-18.41)) / (-5.21 - (-18.41))
    return normalized_value

def find_max_base_pairs(string):
    max_length = 0
    current_length = 0
    max_start_index = -1
    current_start_index = -1
    current_char = ''

    for i, char in enumerate(string):
        if char in '()':
            if char == current_char:
                current_length += 1
            else:
                current_char = char
                current_length = 1
                current_start_index = i
        else:
            current_length = 0

        if current_length > max_length:
            max_length = current_length
            max_start_index = current_start_index

    return max_start_index, max_length

def find_max_unpaired(string):
    max_length = 0
    current_length = 0
    max_start_index = -1
    current_start_index = -1
    current_char = ''

    for i, char in enumerate(string):
        if char in '.':
            if char == current_char:
                current_length += 1
            else:
                current_char = char
                current_length = 1
                current_start_index = i
        else:
            current_length = 0

        if current_length > max_length:
            max_length = current_length
            max_start_index = current_start_index

    return max_start_index, max_length

def detect_5_overhang(input_string):

    current_length = 0
    start_index = None
    for i, char in enumerate(input_string):
        if char == '.':
            if current_length == 0:
                start_index = i
            current_length += 1
        else:
            if current_length > 0:  # Found a sequence of dots (either single or multiple)
                return (start_index, current_length)
            current_length = 0

    # Check if the loop ended with a sequence of dots
    if current_length > 0:
        return (start_index, current_length)

    return (None, 0)

def detect_3_overhang(input_string):

    current_length = 0
    start_index = None
    last_start_index = None
    last_length = 0

    for i, char in enumerate(input_string):
        if char == '.':
            if current_length == 0:
                start_index = i
            current_length += 1
        else:
            if current_length > 0:  # Found a sequence of dots
                last_start_index = start_index
                last_length = current_length
                current_length = 0

    # Check if the loop ended with a sequence of dots
    if current_length > 0:
        last_start_index = start_index
        last_length = current_length

    if last_start_index is not None:
        return (last_start_index, last_length)

    return (None, 0)

def detect_5_paired(input_string):

    current_length = 0
    start_index = None
    for i, char in enumerate(input_string):
        if char == '(' or char == ')':
            if current_length == 0:
                start_index = i
            current_length += 1
        else:
            if current_length > 0:  # Found a sequence of dots (either single or multiple)
                return (start_index, current_length)
            current_length = 0

    # Check if the loop ended with a sequence of dots
    if current_length > 0:
        return (start_index, current_length)

    return (None, 0)

def detect_3_paired(input_string):

    current_length = 0
    start_index = None
    last_start_index = None
    last_length = 0

    for i, char in enumerate(input_string):
        if char == '(' or char == ')':
            if current_length == 0:
                start_index = i
            current_length += 1
        else:
            if current_length > 0:  # Found a sequence of dots
                last_start_index = start_index
                last_length = current_length
                current_length = 0

    # Check if the loop ended with a sequence of dots
    if current_length > 0:
        last_start_index = start_index
        last_length = current_length

    if last_start_index is not None:
        return (last_start_index, last_length)

    return (None, 0)

def is_all_parens(s):
    for char in s:
        if char not in ("(", ")"):
            return False
    return True

def count_dots(s: str) -> int:
    """Return the number of '.' characters in s."""
    return s.count(".")

def build_energy_features(guide, target):
    energy_array_unit = []

    guide = scaffold + guide
    
    # Compute suboptimal structures and energy
    subopt_structures_guide = subopt(strands=guide, energy_gap=0.01, model=my_model_RNA)  
    subopt_structures_target = subopt(strands=[guide, target], energy_gap=0.01, model=my_model_DNA)
    subopt_structures_ssDNA_target_bh = subopt(strands=target, energy_gap=0.01, model=my_model_DNA)
    
    len_scaffold = len(scaffold)
    len_spacer = TEST_GUIDE_LEN

    # Calculate/compare ensemble_energy_max_paired_guide and ensemble_energy_max_paired_target and ensemble_energy_max_paired_ssDNA_target_bh
    if str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer] == '.' * len_spacer:
        ensemble_energy_max_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = find_max_base_pairs(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_paired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_max_paired_guide = pfunc(strands=[max_paired_seq_guide, RNA_reverse_complement(max_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_max_paired_guide = partition_function_max_paired_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer] == '.' * len_spacer:
        ensemble_energy_max_paired_target = 0
    else:
        max_start_index_target, max_length_target = find_max_base_pairs(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_paired_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_max_paired_seq_target = RNA_to_DNA(max_paired_seq_target)
        partition_function_max_paired_target = pfunc(strands=[DNA_max_paired_seq_target, DNA_reverse_complement(DNA_max_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_max_paired_target = partition_function_max_paired_target[1]

    if str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer] == '.' * len_spacer:
        ensemble_energy_max_paired_ssDNA_target_bh = 0
    else:
        max_start_index_ssDNA_target_bh, max_length_ssDNA_target_bh = find_max_base_pairs(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer])
        max_paired_seq_ssDNA_target_bh = guide[max_start_index_ssDNA_target_bh:max_start_index_ssDNA_target_bh+max_length_ssDNA_target_bh]
        DNA_max_paired_seq_ssDNA_target_bh = RNA_to_DNA(max_paired_seq_ssDNA_target_bh)
        partition_function_max_paired_ssDNA_target_bh = pfunc(strands=[DNA_max_paired_seq_ssDNA_target_bh, DNA_reverse_complement(DNA_max_paired_seq_ssDNA_target_bh)], model=my_model_DNA)
        ensemble_energy_max_paired_ssDNA_target_bh = partition_function_max_paired_ssDNA_target_bh[1]

    # Calculate/compare ensemble_energy_max_unpaired_guide and ensemble_energy_max_unpaired_target and ensemble_energy_max_unpaired_ssDNA_target_bh
    if is_all_parens(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer]) == True:
        ensemble_energy_max_unpaired_guide = 0
    else:
        max_start_index_guide, max_length_guide = find_max_unpaired(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_unpaired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_max_unpaired_guide = pfunc(strands=[max_unpaired_seq_guide, RNA_reverse_complement(max_unpaired_seq_guide)], model=my_model_RNA)
        ensemble_energy_max_unpaired_guide = partition_function_max_unpaired_guide[1]

    if is_all_parens(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer]) == True:
        ensemble_energy_max_unpaired_target = 0
    else:
        max_start_index_target, max_length_target = find_max_unpaired(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_unpaired_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_max_unpaired_seq_target = RNA_to_DNA(max_unpaired_seq_target)
        partition_function_max_unpaired_target = pfunc(strands=[DNA_max_unpaired_seq_target, DNA_reverse_complement(DNA_max_unpaired_seq_target)], model=my_model_DNA)
        ensemble_energy_max_unpaired_target = partition_function_max_unpaired_target[1]

    if is_all_parens(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer]) == True:
        ensemble_energy_max_unpaired_ssDNA_target_bh = 0
    else:
        max_start_index_ssDNA_target_bh, max_length_ssDNA_target_bh = find_max_unpaired(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer])
        max_unpaired_seq_ssDNA_target_bh = guide[max_start_index_ssDNA_target_bh:max_start_index_ssDNA_target_bh+max_length_ssDNA_target_bh]
        DNA_max_unpaired_seq_ssDNA_target_bh = RNA_to_DNA(max_unpaired_seq_ssDNA_target_bh)
        partition_function_max_unpaired_ssDNA_target_bh = pfunc(strands=[DNA_max_unpaired_seq_ssDNA_target_bh, DNA_reverse_complement(DNA_max_unpaired_seq_ssDNA_target_bh)], model=my_model_DNA)
        ensemble_energy_max_unpaired_ssDNA_target_bh = partition_function_max_unpaired_ssDNA_target_bh[1]
    
    # Calculate/compare ensemble_energy_PAM_distal_overhang_guide and ensemble_energy_PAM_distal_overhang_target and ensemble_energy_PAM_distal_overhang_ssDNA_target_bh
    if str(subopt_structures_guide[0].structure)[len_scaffold+len_spacer-1] != '.':
        ensemble_energy_PAM_distal_overhang_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_3_overhang(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_distal_overhang_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_PAM_distal_overhang_guide = pfunc(strands=[max_PAM_distal_overhang_seq_guide, RNA_reverse_complement(max_PAM_distal_overhang_seq_guide)], model=my_model_RNA)
        ensemble_energy_PAM_distal_overhang_guide = partition_function_PAM_distal_overhang_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold+len_spacer-1] != '.':
        ensemble_energy_PAM_distal_overhang_target = 0
    else:
        max_start_index_target, max_length_target = detect_3_overhang(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_distal_overhang_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_PAM_distal_overhang_seq_target = RNA_to_DNA(max_PAM_distal_overhang_seq_target)
        partition_function_PAM_distal_overhang_target = pfunc(strands=[DNA_PAM_distal_overhang_seq_target, DNA_reverse_complement(DNA_PAM_distal_overhang_seq_target)], model=my_model_DNA)
        ensemble_energy_PAM_distal_overhang_target = partition_function_PAM_distal_overhang_target[1]

    if str(subopt_structures_ssDNA_target_bh[0].structure)[0] != '.':
        ensemble_energy_PAM_distal_overhang_ssDNA_target_bh = 0
    else:
        max_start_index_ssDNA_target_bh, max_length_ssDNA_target_bh = detect_5_overhang(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer])
        max_PAM_distal_overhang_seq_ssDNA_target_bh = guide[max_start_index_ssDNA_target_bh:max_start_index_ssDNA_target_bh+max_length_ssDNA_target_bh]
        DNA_PAM_distal_overhang_seq_ssDNA_target_bh = RNA_to_DNA(max_PAM_distal_overhang_seq_ssDNA_target_bh)
        partition_function_PAM_distal_overhang_ssDNA_target_bh = pfunc(strands=[DNA_PAM_distal_overhang_seq_ssDNA_target_bh, DNA_reverse_complement(DNA_PAM_distal_overhang_seq_ssDNA_target_bh)], model=my_model_DNA)
        ensemble_energy_PAM_distal_overhang_ssDNA_target_bh = partition_function_PAM_distal_overhang_ssDNA_target_bh[1]
    
    # Calculate/compare ensemble_energy_PAM_proximal_overhang_guide and ensemble_energy_PAM_proximal_overhang_target and ensemble_energy_PAM_proximal_overhang_ssDNA_target_bh
    if str(subopt_structures_guide[0].structure)[len_scaffold] != '.':
        ensemble_energy_PAM_proximal_overhang_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_5_overhang(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_proximal_overhang_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_PAM_proximal_overhang_guide = pfunc(strands=[max_PAM_proximal_overhang_seq_guide, RNA_reverse_complement(max_PAM_proximal_overhang_seq_guide)], model=my_model_RNA)
        ensemble_energy_PAM_proximal_overhang_guide = partition_function_PAM_proximal_overhang_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold] != '.':
        ensemble_energy_PAM_proximal_overhang_target = 0
    else:
        max_start_index_target, max_length_target = detect_5_overhang(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_proximal_overhang_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_PAM_proximal_overhang_seq_target = RNA_to_DNA(max_PAM_proximal_overhang_seq_target)
        partition_function_PAM_proximal_overhang_target = pfunc(strands=[DNA_PAM_proximal_overhang_seq_target, DNA_reverse_complement(DNA_PAM_proximal_overhang_seq_target)], model=my_model_DNA)
        ensemble_energy_PAM_proximal_overhang_target = partition_function_PAM_proximal_overhang_target[1]

    if str(subopt_structures_ssDNA_target_bh[0].structure)[len_spacer-1] != '.':
        ensemble_energy_PAM_proximal_overhang_ssDNA_target_bh = 0
    else:
        max_start_index_ssDNA_target_bh, max_length_ssDNA_target_bh = detect_3_overhang(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer])
        max_PAM_proximal_overhang_seq_ssDNA_target_bh = guide[max_start_index_ssDNA_target_bh:max_start_index_ssDNA_target_bh+max_length_ssDNA_target_bh]
        DNA_PAM_proximal_overhang_seq_ssDNA_target_bh = RNA_to_DNA(max_PAM_proximal_overhang_seq_ssDNA_target_bh)
        partition_function_PAM_proximal_overhang_ssDNA_target_bh = pfunc(strands=[DNA_PAM_proximal_overhang_seq_ssDNA_target_bh, DNA_reverse_complement(DNA_PAM_proximal_overhang_seq_ssDNA_target_bh)], model=my_model_DNA)
        ensemble_energy_PAM_proximal_overhang_ssDNA_target_bh = partition_function_PAM_proximal_overhang_ssDNA_target_bh[1]

    # Calculate/compare ensemble_energy_PAM_distal_paired_guide and ensemble_energy_PAM_distal_paired_target and ensemble_energy_PAM_distal_paired_ssDNA_target_bh
    if str(subopt_structures_guide[0].structure)[len_scaffold+len_spacer-1] == '.':
        ensemble_energy_PAM_distal_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_3_paired(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_distal_paired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_PAM_distal_paired_guide = pfunc(strands=[max_PAM_distal_paired_seq_guide, RNA_reverse_complement(max_PAM_distal_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_PAM_distal_paired_guide = partition_function_PAM_distal_paired_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold+len_spacer-1] == '.':
        ensemble_energy_PAM_distal_paired_target = 0
    else:
        max_start_index_target, max_length_target = detect_3_paired(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_distal_paired_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_PAM_distal_paired_seq_target = RNA_to_DNA(max_PAM_distal_paired_seq_target)
        partition_function_PAM_distal_paired_target = pfunc(strands=[DNA_PAM_distal_paired_seq_target, DNA_reverse_complement(DNA_PAM_distal_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_PAM_distal_paired_target = partition_function_PAM_distal_paired_target[1]

    if str(subopt_structures_ssDNA_target_bh[0].structure)[0] == '.':
        ensemble_energy_PAM_distal_paired_ssDNA_target_bh = 0
    else:
        max_start_index_ssDNA_target_bh, max_length_ssDNA_target_bh = detect_5_paired(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer])
        max_PAM_distal_paired_seq_ssDNA_target_bh = guide[max_start_index_ssDNA_target_bh:max_start_index_ssDNA_target_bh+max_length_ssDNA_target_bh]
        DNA_PAM_distal_paired_seq_ssDNA_target_bh = RNA_to_DNA(max_PAM_distal_paired_seq_ssDNA_target_bh)
        partition_function_PAM_distal_paired_ssDNA_target_bh = pfunc(strands=[DNA_PAM_distal_paired_seq_ssDNA_target_bh, DNA_reverse_complement(DNA_PAM_distal_paired_seq_ssDNA_target_bh)], model=my_model_DNA)
        ensemble_energy_PAM_distal_paired_ssDNA_target_bh = partition_function_PAM_distal_paired_ssDNA_target_bh[1]
    
    # Calculate/compare ensemble_energy_PAM_proximal_paired_guide and ensemble_energy_PAM_proximal_paired_target and ensemble_energy_PAM_proximal_paired_ssDNA_target_bh
    if str(subopt_structures_guide[0].structure)[len_scaffold] == '.':
        ensemble_energy_PAM_proximal_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_5_paired(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_proximal_paired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_PAM_proximal_paired_guide = pfunc(strands=[max_PAM_proximal_paired_seq_guide, RNA_reverse_complement(max_PAM_proximal_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_PAM_proximal_paired_guide = partition_function_PAM_proximal_paired_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold] == '.':
        ensemble_energy_PAM_proximal_paired_target = 0
    else:
        max_start_index_target, max_length_target = detect_5_paired(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_PAM_proximal_paired_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_PAM_proximal_paired_seq_target = RNA_to_DNA(max_PAM_proximal_paired_seq_target)
        partition_function_PAM_proximal_paired_target = pfunc(strands=[DNA_PAM_proximal_paired_seq_target, DNA_reverse_complement(DNA_PAM_proximal_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_PAM_proximal_paired_target = partition_function_PAM_proximal_paired_target[1]

    if str(subopt_structures_ssDNA_target_bh[0].structure)[len_spacer-1] == '.':
        ensemble_energy_PAM_proximal_paired_ssDNA_target_bh = 0
    else:
        max_start_index_ssDNA_target_bh, max_length_ssDNA_target_bh = detect_3_paired(str(subopt_structures_ssDNA_target_bh[0].structure)[0:len_spacer])
        max_PAM_proximal_paired_seq_ssDNA_target_bh = guide[max_start_index_ssDNA_target_bh:max_start_index_ssDNA_target_bh+max_length_ssDNA_target_bh]
        DNA_PAM_proximal_paired_seq_ssDNA_target_bh = RNA_to_DNA(max_PAM_proximal_paired_seq_ssDNA_target_bh)
        partition_function_PAM_proximal_paired_ssDNA_target_bh = pfunc(strands=[DNA_PAM_proximal_paired_seq_ssDNA_target_bh, DNA_reverse_complement(DNA_PAM_proximal_paired_seq_ssDNA_target_bh)], model=my_model_DNA)
        ensemble_energy_PAM_proximal_paired_ssDNA_target_bh = partition_function_PAM_proximal_paired_ssDNA_target_bh[1]
    
    # Calculate/compare target seed region free energy 
    seed_region = target[14:20]
    subopt_structures_seed = subopt(strands=[seed_region, DNA_reverse_complement(seed_region)], energy_gap=6, model=my_model_DNA)
    seed_energy = subopt_structures_seed[0].energy

    # Calculate/compare target middle region free energy 
    middle_region = target[7:14]
    subopt_structures_middle = subopt(strands=[middle_region, DNA_reverse_complement(middle_region)], energy_gap=6, model=my_model_DNA)
    middle_energy = subopt_structures_middle[0].energy

    # Calculate/compare target distal region free energy 
    distal_region = target[0:7]
    subopt_structures_distal = subopt(strands=[distal_region, DNA_reverse_complement(distal_region)], energy_gap=6, model=my_model_DNA)
    distal_energy = subopt_structures_distal[0].energy

    # Detect local energy for consecutive 3 bases in target
    min_ensemble_energy_selected_bases_3 = np.inf
    max_ensemble_energy_selected_bases_3 = -np.inf
    
    for k in range (0, len(target)-2):
        selected_bases_spacer = guide[len_scaffold+k:len_scaffold+3+k]
        selected_bases_target = target[len_spacer-3-k:len_spacer-k]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            ensemble_energy_selected_bases = 2.6148611689341172
        if ensemble_energy_selected_bases > max_ensemble_energy_selected_bases_3:
            max_ensemble_energy_selected_bases_3 = ensemble_energy_selected_bases
        if ensemble_energy_selected_bases < min_ensemble_energy_selected_bases_3:
            min_ensemble_energy_selected_bases_3 = ensemble_energy_selected_bases
    
    # Detect local energy for consecutive 4 bases in target
    min_ensemble_energy_selected_bases_4 = np.inf
    max_ensemble_energy_selected_bases_4 = -np.inf
    
    for k in range (0, len(target)-3):
        selected_bases_spacer = guide[len_scaffold+k:len_scaffold+4+k]
        selected_bases_target = target[len_spacer-4-k:len_spacer-k]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            ensemble_energy_selected_bases = 2.6148611689341172
        if ensemble_energy_selected_bases > max_ensemble_energy_selected_bases_4:
            max_ensemble_energy_selected_bases_4 = ensemble_energy_selected_bases
        if ensemble_energy_selected_bases < min_ensemble_energy_selected_bases_4:
            min_ensemble_energy_selected_bases_4 = ensemble_energy_selected_bases
    
    # Detect local energy for consecutive 5 bases in target
    min_ensemble_energy_selected_bases_5 = np.inf
    max_ensemble_energy_selected_bases_5 = -np.inf
    
    for k in range (0, len(target)-4):
        selected_bases_spacer = guide[len_scaffold+k:len_scaffold+5+k]
        selected_bases_target = target[len_spacer-5-k:len_spacer-k]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            ensemble_energy_selected_bases = 2.6148611689341172
        if ensemble_energy_selected_bases > max_ensemble_energy_selected_bases_5:
            max_ensemble_energy_selected_bases_5 = ensemble_energy_selected_bases
        if ensemble_energy_selected_bases < min_ensemble_energy_selected_bases_5:
            min_ensemble_energy_selected_bases_5 = ensemble_energy_selected_bases
    
    # Detect local energy for consecutive 6 bases in target
    min_ensemble_energy_selected_bases_6 = np.inf
    max_ensemble_energy_selected_bases_6 = -np.inf
    
    for k in range (0, len(target)-5):
        selected_bases_spacer = guide[len_scaffold+k:len_scaffold+6+k]
        selected_bases_target = target[len_spacer-6-k:len_spacer-k]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            ensemble_energy_selected_bases = 2.6148611689341172
        if ensemble_energy_selected_bases > max_ensemble_energy_selected_bases_6:
            max_ensemble_energy_selected_bases_6 = ensemble_energy_selected_bases
        if ensemble_energy_selected_bases < min_ensemble_energy_selected_bases_6:
            min_ensemble_energy_selected_bases_6 = ensemble_energy_selected_bases
    
    # Detect local energy for consecutive 7 bases in target
    min_ensemble_energy_selected_bases_7 = np.inf
    max_ensemble_energy_selected_bases_7 = -np.inf
    
    for k in range (0, len(target)-6):
        selected_bases_spacer = guide[len_scaffold+k:len_scaffold+7+k]
        selected_bases_target = target[len_spacer-7-k:len_spacer-k]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            ensemble_energy_selected_bases = 2.6148611689341172
        if ensemble_energy_selected_bases > max_ensemble_energy_selected_bases_7:
            max_ensemble_energy_selected_bases_7 = ensemble_energy_selected_bases
        if ensemble_energy_selected_bases < min_ensemble_energy_selected_bases_7:
            min_ensemble_energy_selected_bases_7 = ensemble_energy_selected_bases
    
    # Detect local energy for consecutive 8 bases in target
    min_ensemble_energy_selected_bases_8 = np.inf
    max_ensemble_energy_selected_bases_8 = -np.inf
    
    for k in range (0, len(target)-7):
        selected_bases_spacer = guide[len_scaffold+k:len_scaffold+8+k]
        selected_bases_target = target[len_spacer-8-k:len_spacer-k]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            ensemble_energy_selected_bases = 2.6148611689341172
        if ensemble_energy_selected_bases > max_ensemble_energy_selected_bases_8:
            max_ensemble_energy_selected_bases_8 = ensemble_energy_selected_bases
        if ensemble_energy_selected_bases < min_ensemble_energy_selected_bases_8:
            min_ensemble_energy_selected_bases_8 = ensemble_energy_selected_bases
    
    # Compute the energy of every consecutive 4 base pairs in spacer-target duplex
    # This aims to quantify the effect of homopolymers by detecting local high and low energy regions 
    four_base_consecutive_energy_array = [0 for _ in range(len_spacer-3)]

    d = 4
    for s in range (0, len_spacer-3):
        selected_bases_spacer = guide[len_scaffold+s:len_scaffold+s+d]
        selected_bases_target = target[len_spacer-s-d:len_spacer-s]
        partition_function_selected_bases = pfunc(strands=[selected_bases_spacer, selected_bases_target], model=my_model_RNA)
        ensemble_energy_selected_bases = partition_function_selected_bases[1]
        if ensemble_energy_selected_bases == np.inf:
            four_base_consecutive_energy_array[s] = 2.6148611689341172
        else:
            four_base_consecutive_energy_array[s] = ensemble_energy_selected_bases
    
    # Calculate ensemble defect of crRNA-target duplex towards its target structure, which is the fully binding state of crRNA and target duplex   
    ensemble_defect_duplex_array = [0 for _ in range(len_spacer-1)]

    for d in range (0, len_spacer-3):
        for s in range (2, 5):
            guide_segment = guide[len_scaffold+d:len_scaffold+d+s]
            target_segment = target[len_spacer-d-s:len_spacer-d]
            subopt_structures_segment = subopt(strands=[guide_segment, target_segment], energy_gap=0.01, model=my_model_RNA)
            if subopt_structures_segment != []:
                duplex_binding = str(subopt_structures_segment[0].structure)
                ensemble_defect_duplex_ah = count_dots(duplex_binding)/2
            else:
                ensemble_defect_duplex_ah = s
            ensemble_defect_duplex_array[d] += ensemble_defect_duplex_ah

    for d in range (len_spacer-3, len_spacer-1):
        for s in range (2, len_spacer+1-d):
            guide_segment = guide[len_scaffold+d:len_scaffold+d+s]
            target_segment = target[len_spacer-d-s:len_spacer-d]
            subopt_structures_segment = subopt(strands=[guide_segment, target_segment], energy_gap=0.01, model=my_model_RNA)
            if subopt_structures_segment != []:
                duplex_binding = str(subopt_structures_segment[0].structure)
                ensemble_defect_duplex_ah = count_dots(duplex_binding)/2
            else:
                ensemble_defect_duplex_ah = s
            ensemble_defect_duplex_array[d] += ensemble_defect_duplex_ah

    # Calculate the energy of base pairings at incorrect positions of crRNA-target duplex 
    ensemble_defect_energy_duplex_array = [0 for _ in range(len_spacer-2)]

    s = 2
    duplex_fully_binding = "(" * s + "+" + ")" * s
    for guide_start in range (0, len_spacer-4):
        guide_segment = guide[len_scaffold+guide_start:len_scaffold+guide_start+s]
        for target_start in range (len_spacer-5-guide_start, len_spacer-2-guide_start):
            target_segment = target[target_start:target_start+s]    
            subopt_structures_pairing = subopt(strands=[guide_segment, target_segment], energy_gap=0.01, model=my_model_RNA)
            # Compute suboptimal structures and energy of base pairings at incorrect positions
            if subopt_structures_pairing != []:
                if str(subopt_structures_pairing[0].structure) == duplex_fully_binding:
                    if subopt_structures_pairing[0].energy >= 0:
                        ensemble_defect_energy_duplex_array[guide_start] += subopt_structures_pairing[0].energy
                    else:
                        ensemble_defect_energy_duplex_array[guide_start] -= subopt_structures_pairing[0].energy

    for guide_start in range (len_spacer-4, len_spacer-2):
        guide_segment = guide[len_scaffold+guide_start:len_scaffold+guide_start+s]
        for target_start in range (0, len_spacer-2-guide_start):
            target_segment = target[target_start:target_start+s]    
            subopt_structures_pairing = subopt(strands=[guide_segment, target_segment], energy_gap=0.01, model=my_model_RNA)
            # Compute suboptimal structures and energy of base pairings at incorrect positions
            if subopt_structures_pairing != []:
                if str(subopt_structures_pairing[0].structure) == duplex_fully_binding:
                    if subopt_structures_pairing[0].energy >= 0:
                        ensemble_defect_energy_duplex_array[guide_start] += subopt_structures_pairing[0].energy
                    else:
                        ensemble_defect_energy_duplex_array[guide_start] -= subopt_structures_pairing[0].energy


    energy_array_unit = [normalize_guide(subopt_structures_guide[0].energy + 3.47)]
    energy_array_unit.append(normalize_guide_conse(ensemble_energy_max_paired_guide))
    energy_array_unit.append(normalize_guide_conse_unpaired(ensemble_energy_max_unpaired_guide))
    energy_array_unit.append(normalize_guide_overhang(ensemble_energy_PAM_proximal_overhang_guide))
    energy_array_unit.append(normalize_guide_overhang(ensemble_energy_PAM_distal_overhang_guide))
    energy_array_unit.append(normalize_guide_paired(ensemble_energy_PAM_proximal_paired_guide))
    energy_array_unit.append(normalize_guide_paired(ensemble_energy_PAM_distal_paired_guide))
    
    energy_array_unit.append(normalize_target(subopt_structures_target[0].energy))
    energy_array_unit.append(normalize_target_conse(ensemble_energy_max_paired_target))
    energy_array_unit.append(normalize_target_conse_unpaired(ensemble_energy_max_unpaired_target))
    energy_array_unit.append(normalize_target_overhang(ensemble_energy_PAM_proximal_overhang_target))
    energy_array_unit.append(normalize_target_overhang(ensemble_energy_PAM_distal_overhang_target))
    energy_array_unit.append(normalize_target_paired(ensemble_energy_PAM_proximal_paired_target))
    energy_array_unit.append(normalize_target_paired(ensemble_energy_PAM_distal_paired_target))

    energy_array_unit.append(normalize_ssDNA_target_bh(subopt_structures_ssDNA_target_bh[0].energy))
    energy_array_unit.append(normalize_ssDNA_target_bh_conse(ensemble_energy_max_paired_ssDNA_target_bh))
    energy_array_unit.append(normalize_ssDNA_target_bh_conse_unpaired(ensemble_energy_max_unpaired_ssDNA_target_bh))
    energy_array_unit.append(normalize_ssDNA_target_bh_overhang(ensemble_energy_PAM_proximal_overhang_ssDNA_target_bh))
    energy_array_unit.append(normalize_ssDNA_target_bh_overhang(ensemble_energy_PAM_distal_overhang_ssDNA_target_bh))
    energy_array_unit.append(normalize_ssDNA_target_bh_paired(ensemble_energy_PAM_proximal_paired_ssDNA_target_bh))
    energy_array_unit.append(normalize_ssDNA_target_bh_paired(ensemble_energy_PAM_distal_paired_ssDNA_target_bh))

    energy_array_unit.append(normalize_seed(seed_energy))
    energy_array_unit.append(normalize_middle(middle_energy))
    energy_array_unit.append(normalize_distal(distal_energy))

    energy_array_unit.append(normalize_target_conse_3(max_ensemble_energy_selected_bases_3))
    energy_array_unit.append(normalize_target_conse_3(min_ensemble_energy_selected_bases_3))
    energy_array_unit.append(normalize_target_conse_4(max_ensemble_energy_selected_bases_4))
    energy_array_unit.append(normalize_target_conse_4(min_ensemble_energy_selected_bases_4))
    energy_array_unit.append(normalize_target_conse_5(max_ensemble_energy_selected_bases_5))
    energy_array_unit.append(normalize_target_conse_5(min_ensemble_energy_selected_bases_5))
    energy_array_unit.append(normalize_target_conse_6(max_ensemble_energy_selected_bases_6))
    energy_array_unit.append(normalize_target_conse_6(min_ensemble_energy_selected_bases_6))
    energy_array_unit.append(normalize_target_conse_7(max_ensemble_energy_selected_bases_7))
    energy_array_unit.append(normalize_target_conse_7(min_ensemble_energy_selected_bases_7))
    energy_array_unit.append(normalize_target_conse_8(max_ensemble_energy_selected_bases_8))
    energy_array_unit.append(normalize_target_conse_8(min_ensemble_energy_selected_bases_8))

    for s in range (0, len_spacer-3):
        energy_array_unit.append(normalize_target_conse_4(four_base_consecutive_energy_array[s]))

    for s in range (0, 27-(len_spacer-3)):
        energy_array_unit.append(0)

    for d in range (0, len_spacer-1):
        energy_array_unit.append(ensemble_defect_duplex_array[d])

    for s in range (0, 29-(len_spacer-1)):
        energy_array_unit.append(0)

    for s in range (0, len_spacer-2):
        energy_array_unit.append(ensemble_defect_energy_duplex_array[s])

    for s in range (0, 28-(len_spacer-2)):
        energy_array_unit.append(0)

    arr = np.array(energy_array_unit, dtype=np.float32)  
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# 4. Graph features for GNN
# -----------------------------
def matrix_to_edge_index(prob_matrix):
    edges_source = []
    edges_target = []
    L = len(prob_matrix)
    for i in range(L):
        for j in range(i + 1, L):  # only i < j
            if prob_matrix[i][j] > 0.01:
                edges_source.append(i)
                edges_target.append(j)
    return [edges_source, edges_target]

def generate_edge_features(edge_index, prob_matrix):
    if not edge_index:
        return []
    feats = []
    for i in range(len(edge_index[0])):
        src = edge_index[0][i]; tgt = edge_index[1][i]
        feats.append([prob_matrix[src][tgt]])
    return feats

def collapse_to_last_k_plus_1_sum(prob_matrix: np.ndarray, last_k: int = TEST_GUIDE_LEN) -> np.ndarray:
    """
    last_k refers to the spacer length
    Collapse an LxL base-pair probability matrix into (last_k + 1) x (last_k + 1):
      - Rows/cols last_k..38: original last_k_base..39
      - Row/col 0: last_k: an 'outside' super-node representing bases <= last_k
    For each i in [last_k..38], the edge prob to the super-node is the
    SUM of probabilities of pairing to bases <= last_k.
    """
    L = prob_matrix.shape[0]
    K = last_k
    new_size = K + 1
    newP = np.zeros((new_size, new_size), dtype=float)

    newP[:K, :K] = prob_matrix[(L-K):, (L-K):]

    if L > K:
        # Outside block (0..K-1)
        outside_block = prob_matrix[(L-K):L, :(L-K)]  # shape: K x (L-K)
        # Sum of probabilities across outside bases
        p_sum = np.sum(outside_block, axis=1)
        # Fill symmetric connections to the super-node K
        newP[:K, K] = p_sum
        newP[K, :K] = p_sum

    # Leave newP[K, K] as 0
    return newP

def slice_and_renumber_duplex(prob_matrix, len_guide, len_target, k=TEST_GUIDE_LEN):
    """
    k refers to the spacer length
    Keep only the last k positions of the guide and the last k positions of the target,
    Returns a (kg+kt) x (kg+kt) reduced/renumbered matrix.
    """
    L = len_guide + len_target
    assert prob_matrix.shape == (L, L), "prob_matrix size mismatch."

    kg = min(k, len_guide)
    kt = min(k, len_target)

    # Original indices: guide = [0..len_guide-1], target = [len_guide..L-1]
    sel_g = np.arange(len_guide - kg, len_guide)  # last k of guide
    sel_t = np.arange(L - kt, L)                  # last k of target
    S = np.concatenate([sel_g, sel_t])

    # Extract submatrix for these rows/cols
    P = prob_matrix[np.ix_(S, S)].copy()

    # Reverse target rows/cols
    P[kg:, :] = P[kg:, :][::-1, :]
    P[:, kg:] = P[:, kg:][:, ::-1]

    return P

def build_guide_graph(guide):
    """
    crRNA graph:
      - use RNA self-pairing probability matrix
      - collapse to last_k + 1 nodes
      - only probability edge feature
      - use zero node features as placeholder
    """
    last_k = TEST_GUIDE_LEN
    prob_full = pairs(strands=guide, model=my_model_RNA).to_array()
    prob_collapsed = collapse_to_last_k_plus_1_sum(prob_full, last_k=last_k)

    edge_index_list = matrix_to_edge_index(prob_collapsed)
    edge_features = generate_edge_features(edge_index_list, prob_collapsed)

    n_nodes = last_k + 1
    x = torch.zeros((n_nodes, 4), dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_target_bh_graph(target):
    """
    target BH graph:
      - DNA self-pairing
      - probability-only edge features
      - zero node features
    """
    prob_matrix = pairs(strands=target, model=my_model_DNA).to_array()
    prob_matrix = prob_matrix[::-1, ::-1]

    edge_index_list = matrix_to_edge_index(prob_matrix)
    edge_features = generate_edge_features(edge_index_list, prob_matrix)

    n_nodes = prob_matrix.shape[0]
    x = torch.zeros((n_nodes, 4), dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_duplex_graph(guide, target):
    """
    duplex graph:
      - DNA/DNA (or whatever your chosen model is)
      - keep guide last k and target first k
      - probability-only edge features
      - zero node features
    """
    k = TEST_GUIDE_LEN
    prob_matrix = pairs(strands=[guide, target], model=my_model_DNA).to_array()

    len_g = len(guide)
    len_t = len(target)

    updated_P = slice_and_renumber_duplex(prob_matrix, len_g, len_t, k=k)

    edge_index_list = matrix_to_edge_index(updated_P)
    edge_features = generate_edge_features(edge_index_list, updated_P)

    n_nodes = updated_P.shape[0]
    x = torch.zeros((n_nodes, 4), dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
