# features.py
import numpy as np
import torch
from torch_geometric.data import Data

from nupack import Model, pairs
from nupack import *

TEST_GUIDE_LEN = 21
TEST_TARGET_LEN = 21
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

# -----------------------------
# 1. Sequence features: x1
# shape should be (1, 84, 4)
# -----------------------------
def build_sequence_features(guide, target):
    guide = scaffold + guide
    seq_array_unit = encode_sequence(guide) + encode_sequence(target) + encode_sequence(DNA_reverse_complement(target))
    arr = np.array(seq_array_unit, dtype=np.float32).reshape(84, 4)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# 2. Structure features: x2
# shape should be (1, 168, 3)
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
    subopt_structures_target_bh = subopt(strands=[target, DNA_reverse_complement(target)], energy_gap=0.1, model=my_model_DNA)
    subopt_structures_hybrid_ah = subopt(strands=[guide, target], energy_gap=0.1, model=my_model_DNA)
    subopt_structures_target_ah = subopt(strands=DNA_reverse_complement(target), energy_gap=0.1, model=my_model_DNA)

    # Guide structure before hybridization
    for j in range (0, len(guide)):
        
        try:
            if str(subopt_structures_guide_bh[0].structure)[j] == '.':
                to_be_added = ([1, 0, 0])
            elif str(subopt_structures_guide_bh[0].structure)[j] == '(':
                to_be_added = ([0, 1, 0])
            elif str(subopt_structures_guide_bh[0].structure)[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
                
        except IndexError:
            to_be_added = ([0, 0, 0])

        struct_array_unit.append(to_be_added)
        
    # Target structure before hybridization 
    target_bh_1 = get_segments_interrupted_by_plus(str(subopt_structures_target_bh[0].structure))[0]
    target_bh_2 = get_segments_interrupted_by_plus(str(subopt_structures_target_bh[0].structure))[1]

    for j in range (0, len(target)):
        
        try:
            if target_bh_1[j] == '.':
                to_be_added = ([1, 0, 0])
            elif target_bh_1[j] == '(':
                to_be_added = ([0, 1, 0])
            elif target_bh_1[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)

    for j in range (0, len(target)):
        
        try:
            if target_bh_2[j] == '.':
                to_be_added = ([1, 0, 0])
            elif target_bh_2[j] == '(':
                to_be_added = ([0, 1, 0])
            elif target_bh_2[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])

        struct_array_unit.append(to_be_added)

    # Hybrid structure after hybridization 
    hybrid_ah_1 = get_segments_interrupted_by_plus(str(subopt_structures_hybrid_ah[0].structure))[0]
    hybrid_ah_2 = get_segments_interrupted_by_plus(str(subopt_structures_hybrid_ah[0].structure))[1]

    for j in range (0, len(guide)):
        
        try:
            if hybrid_ah_1[j] == '.':
                to_be_added = ([1, 0, 0])
            elif hybrid_ah_1[j] == '(':
                to_be_added = ([0, 1, 0])
            elif hybrid_ah_1[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)
     
    for j in range (0, len(target)):
        
        try:
            if hybrid_ah_2[j] == '.':
                to_be_added = ([1, 0, 0])
            elif hybrid_ah_2[j] == '(':
                to_be_added = ([0, 1, 0])
            elif hybrid_ah_2[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])

        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)

    # Target structure after hybridization
    for j in range (0, len(target)):
        
        try:
            if str(subopt_structures_target_ah[0].structure)[j] == '.':
                to_be_added = ([1, 0, 0])
            elif str(subopt_structures_target_ah[0].structure)[j] == '(':
                to_be_added = ([0, 1, 0])
            elif str(subopt_structures_target_ah[0].structure)[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
                
        except IndexError:
            to_be_added = ([0, 0, 0])

        struct_array_unit.append(to_be_added)

    arr = np.array(struct_array_unit, dtype=np.float32).reshape(168, 3)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# 3. Energy features: x3
# shape should be (1, 17)
# -----------------------------
def RNA_to_DNA(RNA):
    match = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'T'}
    return ''.join(match.get(base, base) for base in (RNA))

def RNA_reverse_complement(RNA):
    complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(RNA))

def normalize_guide(value):
    normalized_value = (value - (-12)) / (1 - (-12))
    return normalized_value

def normalize_target(value):
    normalized_value = (value - (-38)) / (-13 - (-38))
    return normalized_value

def normalize_guide_conse(value):
    normalized_value = (value - (-28)) / (1 - (-28))
    return normalized_value

def normalize_target_conse(value):
    normalized_value = (value - (-37)) / (-7 - (-37))
    return normalized_value

def normalize_guide_conse_unpaired(value):
    normalized_value = (value - (-47)) / (2 - (-47))
    return normalized_value

def normalize_target_conse_unpaired(value):
    normalized_value = (value - (-5)) / (0 - (-5))
    return normalized_value

def normalize_guide_overhang(value):
    normalized_value = (value - (-47)) / (3 - (-47))
    return normalized_value

def normalize_target_overhang(value):
    normalized_value = (value - (-5)) / (0 - (-5))
    return normalized_value

def normalize_guide_paired(value):
    normalized_value = (value - (-22)) / (3 - (-22))
    return normalized_value

def normalize_target_paired(value):
    normalized_value = (value - (-37)) / (0 - (-37))
    return normalized_value

def normalize_seed(value):
    normalized_value = (value - (-11)) / (-3 - (-11))
    return normalized_value

def normalize_middle(value):
    normalized_value = (value - (-15)) / (-4 - (-15))
    return normalized_value

def normalize_distal(value):
    normalized_value = (value - (-13)) / (-4 - (-13))
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

def build_energy_features(guide, target):
    energy_array_unit = []

    guide = scaffold + guide
    
    # Compute suboptimal structures and energy
    subopt_structures_guide = subopt(strands=guide, energy_gap=0.1, model=my_model_RNA)  
    subopt_structures_target = subopt(strands=[guide, target], energy_gap=0.1, model=my_model_DNA)

    len_scaffold = 21
    len_spacer = 21
        
    # Calculate/compare ensemble_energy_max_paired_guide and ensemble_energy_max_paired_target
    if str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer] == '.' * len_spacer:
        ensemble_energy_max_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = find_max_base_pairs(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_paired_seq_guide = guide[max_start_index_guide+len_scaffold:max_start_index_guide+max_length_guide+len_scaffold]
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

    # Calculate/compare ensemble_energy_max_unpaired_guide and ensemble_energy_max_unpaired_target
    if is_all_parens(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer]) == True:
        ensemble_energy_max_unpaired_guide = 0
    else:
        max_start_index_guide, max_length_guide = find_max_unpaired(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_unpaired_seq_guide = guide[max_start_index_guide+len_scaffold:max_start_index_guide+max_length_guide+len_scaffold]
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

    # Calculate/compare ensemble_energy_5_overhang_guide and ensemble_energy_5_overhang_target
    if str(subopt_structures_guide[0].structure)[len_scaffold] != '.':
        ensemble_energy_5_overhang_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_5_overhang(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_5_overhang_seq_guide = guide[max_start_index_guide+len_scaffold:max_start_index_guide+max_length_guide+len_scaffold]
        partition_function_5_overhang_guide = pfunc(strands=[max_5_overhang_seq_guide, RNA_reverse_complement(max_5_overhang_seq_guide)], model=my_model_RNA)
        ensemble_energy_5_overhang_guide = partition_function_5_overhang_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold] != '.':
        ensemble_energy_5_overhang_target = 0
    else:
        max_start_index_target, max_length_target = detect_5_overhang(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_5_overhang_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_5_overhang_seq_target = RNA_to_DNA(max_5_overhang_seq_target)
        partition_function_5_overhang_target = pfunc(strands=[DNA_5_overhang_seq_target, DNA_reverse_complement(DNA_5_overhang_seq_target)], model=my_model_DNA)
        ensemble_energy_5_overhang_target = partition_function_5_overhang_target[1]
            
    # Calculate/compare ensemble_energy_3_overhang_guide and ensemble_energy_3_overhang_target
    if str(subopt_structures_guide[0].structure)[len_scaffold+len_spacer-1] != '.':
        ensemble_energy_3_overhang_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_3_overhang(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_3_overhang_seq_guide = guide[max_start_index_guide+len_scaffold:max_start_index_guide+max_length_guide+len_scaffold]
        partition_function_3_overhang_guide = pfunc(strands=[max_3_overhang_seq_guide, RNA_reverse_complement(max_3_overhang_seq_guide)], model=my_model_RNA)
        ensemble_energy_3_overhang_guide = partition_function_3_overhang_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold+len_spacer-1] != '.':
        ensemble_energy_3_overhang_target = 0
    else:
        max_start_index_target, max_length_target = detect_3_overhang(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_3_overhang_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_3_overhang_seq_target = RNA_to_DNA(max_3_overhang_seq_target)
        partition_function_3_overhang_target = pfunc(strands=[DNA_3_overhang_seq_target, DNA_reverse_complement(DNA_3_overhang_seq_target)], model=my_model_DNA)
        ensemble_energy_3_overhang_target = partition_function_3_overhang_target[1]
            
    # Calculate/compare ensemble_energy_5_paired_guide and ensemble_energy_5_paired_target
    if str(subopt_structures_guide[0].structure)[len_scaffold] == '.':
        ensemble_energy_5_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_5_paired(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_5_paired_seq_guide = guide[max_start_index_guide+len_scaffold:max_start_index_guide+max_length_guide+len_scaffold]
        partition_function_5_paired_guide = pfunc(strands=[max_5_paired_seq_guide, RNA_reverse_complement(max_5_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_5_paired_guide = partition_function_5_paired_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold] == '.':
        ensemble_energy_5_paired_target = 0
    else:
        max_start_index_target, max_length_target = detect_5_paired(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_5_paired_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_5_paired_seq_target = RNA_to_DNA(max_5_paired_seq_target)
        partition_function_5_paired_target = pfunc(strands=[DNA_5_paired_seq_target, DNA_reverse_complement(DNA_5_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_5_paired_target = partition_function_5_paired_target[1]

    # Calculate/compare ensemble_energy_3_paired_guide and ensemble_energy_3_paired_target
    if str(subopt_structures_guide[0].structure)[len_scaffold+len_spacer-1] == '.':
        ensemble_energy_3_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_3_paired(str(subopt_structures_guide[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_3_paired_seq_guide = guide[max_start_index_guide+len_scaffold:max_start_index_guide+max_length_guide+len_scaffold]
        partition_function_3_paired_guide = pfunc(strands=[max_3_paired_seq_guide, RNA_reverse_complement(max_3_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_3_paired_guide = partition_function_3_paired_guide[1]

    if str(subopt_structures_target[0].structure)[len_scaffold+len_spacer-1] == '.':
        ensemble_energy_3_paired_target = 0
    else:
        max_start_index_target, max_length_target = detect_3_paired(str(subopt_structures_target[0].structure)[len_scaffold:len_scaffold+len_spacer])
        max_3_paired_seq_target = guide[max_start_index_target+len_scaffold:max_start_index_target+max_length_target+len_scaffold]
        DNA_3_paired_seq_target = RNA_to_DNA(max_3_paired_seq_target)
        partition_function_3_paired_target = pfunc(strands=[DNA_3_paired_seq_target, DNA_reverse_complement(DNA_3_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_3_paired_target = partition_function_3_paired_target[1]

    # Calculate/compare target seed region free energy 
    seed_region = target[15:21]
    subopt_structures_seed = subopt(strands=[seed_region, DNA_reverse_complement(seed_region)], energy_gap=6, model=my_model_DNA)
    seed_energy = subopt_structures_seed[0].energy

    # Calculate/compare target middle region free energy 
    middle_region = target[7:15]
    subopt_structures_middle = subopt(strands=[middle_region, DNA_reverse_complement(middle_region)], energy_gap=6, model=my_model_DNA)
    middle_energy = subopt_structures_middle[0].energy

    # Calculate/compare target distal region free energy 
    distal_region = target[0:7]
    subopt_structures_distal = subopt(strands=[distal_region, DNA_reverse_complement(distal_region)], energy_gap=6, model=my_model_DNA)
    distal_energy = subopt_structures_distal[0].energy

    energy_array_unit = [normalize_guide(subopt_structures_guide[0].energy + 3.47)]
    energy_array_unit.append(normalize_guide_conse(ensemble_energy_max_paired_guide))
    energy_array_unit.append(normalize_guide_conse_unpaired(ensemble_energy_max_unpaired_guide))
    energy_array_unit.append(normalize_guide_overhang(ensemble_energy_5_overhang_guide))
    energy_array_unit.append(normalize_guide_overhang(ensemble_energy_3_overhang_guide))
    energy_array_unit.append(normalize_guide_paired(ensemble_energy_5_paired_guide))
    energy_array_unit.append(normalize_guide_paired(ensemble_energy_3_paired_guide))
    
    energy_array_unit.append(normalize_target(subopt_structures_target[0].energy))
    energy_array_unit.append(normalize_target_conse(ensemble_energy_max_paired_target))
    energy_array_unit.append(normalize_target_conse_unpaired(ensemble_energy_max_unpaired_target))
    energy_array_unit.append(normalize_target_overhang(ensemble_energy_5_overhang_target))
    energy_array_unit.append(normalize_target_overhang(ensemble_energy_3_overhang_target))
    energy_array_unit.append(normalize_target_paired(ensemble_energy_5_paired_target))
    energy_array_unit.append(normalize_target_paired(ensemble_energy_3_paired_target))
    
    energy_array_unit.append(normalize_seed(seed_energy))
    energy_array_unit.append(normalize_middle(middle_energy))
    energy_array_unit.append(normalize_distal(distal_energy))

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

def generate_edge_features(edge_index, seq_one_hot, prob_matrix):
    features = []
    if not edge_index:
        return []
    for i in range(len(edge_index[0])):
        src_idx = edge_index[0][i]
        tgt_idx = edge_index[1][i]
        prob_val = prob_matrix[src_idx][tgt_idx]
        
        def decode_nt(one_hot_vec):
            if one_hot_vec[0] == 1:
                return 'A'
            elif one_hot_vec[1] == 1:
                return 'C'
            elif one_hot_vec[2] == 1:
                return 'G'
            elif one_hot_vec[3] == 1:
                return 'T'
            return 'N'
        
        src_char = decode_nt(seq_one_hot[src_idx])
        tgt_char = decode_nt(seq_one_hot[tgt_idx])

        if (src_char == 'A' and tgt_char in ['T','U']) or (src_char in ['T','U'] and tgt_char == 'A'):
            pair_type = 2
        elif (src_char == 'G' and tgt_char in ['T','U']) or (src_char in ['T','U'] and tgt_char == 'G'):
            pair_type = 2
        elif (src_char == 'C' and tgt_char == 'G') or (src_char == 'G' and tgt_char == 'C'):
            pair_type = 3
        
        features.append([pair_type/3, prob_val])
    return features

def build_guide_graph(guide):
    guide = scaffold + guide
    prob_matrix = pairs(strands=guide, model=my_model_RNA).to_array()
    node_features = encode_sequence(guide)
    edge_index_list = matrix_to_edge_index(prob_matrix)
    edge_features = generate_edge_features(edge_index_list, node_features, prob_matrix)

    x = torch.tensor(node_features, dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_target_bh_graph(target):
    duplex_seq = target + DNA_reverse_complement(target)
    prob_matrix = pairs(strands=[target, DNA_reverse_complement(target)], model=my_model_DNA).to_array()
    node_features = encode_sequence(duplex_seq)
    edge_index_list = matrix_to_edge_index(prob_matrix)
    edge_features = generate_edge_features(edge_index_list, node_features, prob_matrix)

    x = torch.tensor(node_features, dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_duplex_graph(guide, target):
    guide = scaffold + guide
    duplex_seq = guide + target
    prob_matrix = pairs(strands=[guide, target], model=my_model_DNA).to_array()
    node_features = encode_sequence(duplex_seq)
    edge_index_list = matrix_to_edge_index(prob_matrix)
    edge_features = generate_edge_features(edge_index_list, node_features, prob_matrix)

    x = torch.tensor(node_features, dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_target_ah_graph(target):
    prob_matrix = pairs(strands=DNA_reverse_complement(target), model=my_model_DNA).to_array()
    node_features = encode_sequence(DNA_reverse_complement(target))
    edge_index_list = matrix_to_edge_index(prob_matrix)
    edge_features = generate_edge_features(edge_index_list, node_features, prob_matrix)

    x = torch.tensor(node_features, dtype=torch.float32)

    if len(edge_index_list[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    