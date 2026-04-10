# features.py
import numpy as np
import torch
from torch_geometric.data import Data

from nupack import Model, pairs
from nupack import *

TEST_GUIDE_LEN = 20
TEST_TARGET_LEN = 47

# Define NUPACK model
my_model_RNA = Model(material='rna', ensemble='nostacking', celsius=37, sodium=0.1, magnesium=0.01)
my_model_DNA = Model(material='dna', ensemble='nostacking', celsius=37, sodium=0.1, magnesium=0.01)

scaffold = "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUUU"
subopt_structures_scaffold = subopt(strands=scaffold, energy_gap=0.1, model=my_model_RNA)

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
# shape should be (1, 231, 4)
# -----------------------------
def build_sequence_features(guide, target, template):
    guide = guide + scaffold + template + '-' * (37 - len(template))
    seq_array_unit = encode_sequence(guide) + encode_sequence(target) + encode_sequence(DNA_reverse_complement(target))
    arr = np.array(seq_array_unit, dtype=np.float32).reshape(231, 4)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# 2. Structure features: x2
# shape should be (1, 705, 3)
# -----------------------------
def DNA_reverse_complement(DNA):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(DNA))

def DNA_to_RNA(DNA):
    match = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'U'}
    return ''.join(match.get(base, base) for base in (DNA))

def get_segments_interrupted_by_plus(input_string: str) -> list:
    # Split the string by the "+" character
    segments = input_string.split("+")
    
    # Remove empty segments if any (e.g., if the string starts or ends with "+")
    segments = [segment for segment in segments if segment]
    
    return segments

def build_structure_features(guide, target, template):
    struct_array_unit = []

    guide = guide + scaffold + template
    target = target[23:43]

    # Compute suboptimal structures and energy
    subopt_structures_guide_bh = subopt(strands=guide, energy_gap=0.1, model=my_model_RNA)  
    subopt_structures_target_bh = subopt(strands=[target, DNA_reverse_complement(target)], energy_gap=0.1, model=my_model_DNA)
    subopt_structures_hybrid_ah = subopt(strands=[guide, target], energy_gap=0.1, model=my_model_DNA)
    subopt_structures_target_ah = subopt(strands=DNA_reverse_complement(target), energy_gap=0.1, model=my_model_DNA)

    new_guide_with_target = DNA_to_RNA(target) + guide
    subopt_structures_triplex_bc = subopt(strands=[new_guide_with_target, DNA_reverse_complement(target)], energy_gap=0.1, model=my_model_DNA)
    subopt_structures_triplex_ac = subopt(strands=[new_guide_with_target, DNA_reverse_complement(target)[0:17]], energy_gap=0.1, model=my_model_DNA)

    # Guide structure before hybridization
    for j in range (0, 137):
        
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

    for j in range (0, 137):
        
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

    # Triplex structure before cleavage
    triplex_bc_1 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_bc[0].structure))[0]
    triplex_bc_2 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_bc[0].structure))[1]

    for j in range (0, 137 + len(target)):
        
        try:
            if triplex_bc_1[j] == '.':
                to_be_added = ([1, 0, 0])
            elif triplex_bc_1[j] == '(':
                to_be_added = ([0, 1, 0])
            elif triplex_bc_1[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)

    for j in range (0, len(target)):

        try:
            if triplex_bc_2[j] == '.':
                to_be_added = ([1, 0, 0])
            elif triplex_bc_2[j] == '(':
                to_be_added = ([0, 1, 0])
            elif triplex_bc_2[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])

        struct_array_unit.append(to_be_added)    

    # Triplex structure after cleavage
    triplex_ac_1 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_ac[0].structure))[0]
    triplex_ac_2 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_ac[0].structure))[1]

    for j in range (0, 137 + len(target)):
        
        try:
            if triplex_ac_1[j] == '.':
                to_be_added = ([1, 0, 0])
            elif triplex_ac_1[j] == '(':
                to_be_added = ([0, 1, 0])
            elif triplex_ac_1[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])
        
        struct_array_unit.append(to_be_added)
     
    for j in range (0, 17):
        
        try:
            if triplex_ac_2[j] == '.':
                to_be_added = ([1, 0, 0])
            elif triplex_ac_2[j] == '(':
                to_be_added = ([0, 1, 0])
            elif triplex_ac_2[j] == ')':
                to_be_added = ([0, 0, 1])
            else:
                to_be_added = ([0, 0, 0])
            
        except IndexError:
             to_be_added = ([0, 0, 0])

        struct_array_unit.append(to_be_added)

    arr = np.array(struct_array_unit, dtype=np.float32).reshape(705, 3)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# 3. Energy features: x3
# shape should be (1, 41)
# -----------------------------
def RNA_to_DNA(RNA):
    match = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'T'}
    return ''.join(match.get(base, base) for base in (RNA))

def RNA_reverse_complement(RNA):
    complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(RNA))

def get_initial_num_dots(s):
    i = 0
    while i < len(s) and s[i] == '.':
        i += 1
    initial_dots = s[:i]
    num_initial_dots = len(initial_dots)
    return num_initial_dots

def normalize_guide(value):
    normalized_value = (value - (-49)) / (0 - (-49))
    return normalized_value

def normalize_target(value):
    normalized_value = (value - (-51)) / (-25 - (-51))
    return normalized_value

def normalize_guide_conse(value):
    normalized_value = (value - (-52)) / (3 - (-52))
    return normalized_value

def normalize_target_conse(value):
    normalized_value = (value - (-34)) / (-17 - (-34))
    return normalized_value

def normalize_guide_conse_ext_bc(value):
    normalized_value = (value - (-52)) / (2 - (-52))
    return normalized_value

def normalize_target_conse_ext_bc(value):
    normalized_value = (value - (-34)) / (0 - (-34))
    return normalized_value

def normalize_target_conse_ext_ac(value):
    normalized_value = (value - (-30)) / (0 - (-30))
    return normalized_value

def normalize_target_conse_RT(value):
    normalized_value = (value - (-26)) / (0 - (-26))
    return normalized_value

def normalize_guide_conse_unpaired(value):
    normalized_value = (value - (-41)) / (3 - (-41))
    return normalized_value

def normalize_target_conse_unpaired(value):
    normalized_value = (value - (-0.5)) / (0 - (-0.5))
    return normalized_value

def normalize_guide_conse_unpaired_ext_bc(value):
    normalized_value = (value - (-40)) / (3 - (-40))
    return normalized_value

def normalize_target_conse_unpaired_ext_bc(value):
    normalized_value = (value - (-27)) / (0 - (-27))
    return normalized_value

def normalize_target_conse_unpaired_ext_ac(value):
    normalized_value = (value - (-27)) / (0 - (-27))
    return normalized_value

def normalize_target_conse_unpaired_RT(value):
    normalized_value = (value - (-35)) / (0 - (-35))
    return normalized_value

def normalize_guide_overhang(value):
    normalized_value = (value - (-41)) / (3 - (-41))
    return normalized_value

def normalize_target_overhang(value):
    normalized_value = (value - (-0.5)) / (0 - (-0.5))
    return normalized_value

def normalize_guide_overhang_ext_bc(value):
    normalized_value = (value - (-40)) / (3 - (-40))
    return normalized_value

def normalize_target_overhang_ext_bc(value):
    normalized_value = (value - (-27)) / (0 - (-27))
    return normalized_value

def normalize_target_overhang_ext_ac(value):
    normalized_value = (value - (-27)) / (0 - (-27))
    return normalized_value

def normalize_target_overhang_RT(value):
    normalized_value = (value - (-26)) / (0 - (-26))
    return normalized_value

def normalize_guide_paired(value):
    normalized_value = (value - (-52)) / (3 - (-52))
    return normalized_value

def normalize_target_paired(value):
    normalized_value = (value - (-34)) / (0 - (-34))
    return normalized_value

def normalize_guide_paired_ext_bc(value):
    normalized_value = (value - (-52)) / (3 - (-52))
    return normalized_value

def normalize_target_paired_ext_bc(value):
    normalized_value = (value - (-34)) / (0 - (-34))
    return normalized_value

def normalize_target_paired_ext_ac(value):
    normalized_value = (value - (-34)) / (0 - (-34))
    return normalized_value

def normalize_target_paired_RT(value):
    normalized_value = (value - (-26)) / (0 - (-26))
    return normalized_value

def normalize_seed(value):
    normalized_value = (value - (-11)) / (-3 - (-11))
    return normalized_value

def normalize_middle(value):
    normalized_value = (value - (-13)) / (-4 - (-13))
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

def build_energy_features(guide, target, template):
    energy_array_unit = []

    guide = guide + scaffold + template
    target = target[23:43]

    # Compute suboptimal structures and energy
    subopt_structures_guide = subopt(strands=guide, energy_gap=0.1, model=my_model_RNA)  
    subopt_structures_target = subopt(strands=[guide, target], energy_gap=0.1, model=my_model_DNA)

    new_guide_with_target = DNA_to_RNA(target) + guide
    subopt_structures_triplex_bc = subopt(strands=[new_guide_with_target, DNA_reverse_complement(target)], energy_gap=0.1, model=my_model_DNA)
    subopt_structures_triplex_ac = subopt(strands=[new_guide_with_target, DNA_reverse_complement(target)[0:17]], energy_gap=0.1, model=my_model_DNA)

    triplex_bc_1 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_bc[0].structure))[0]
    triplex_bc_2 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_bc[0].structure))[1]

    triplex_ac_1 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_ac[0].structure))[0]
    triplex_ac_2 = get_segments_interrupted_by_plus(str(subopt_structures_triplex_ac[0].structure))[1]

    len_scaffold = 80
    len_spacer = 20
    len_extension = len(guide) - len_scaffold - len_spacer

    guide_linear_struct = '.'*20 + str(subopt_structures_scaffold[0].structure) + '.'*len_extension
    dGstruc_guide_linear = structure_energy(strands=guide, structure=guide_linear_struct, model=my_model_RNA)

    #------Energy properties for the spacer region of guide before and after hybridizing to its target------
        
    # Calculate/compare ensemble_energy_max_paired_guide_spacer and ensemble_energy_max_paired_target_spacer
    if str(subopt_structures_guide[0].structure)[0:len_spacer] == '.' * len_spacer:
        ensemble_energy_max_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = find_max_base_pairs(str(subopt_structures_guide[0].structure)[0:len_spacer])
        max_paired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_max_paired_guide = pfunc(strands=[max_paired_seq_guide, RNA_reverse_complement(max_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_max_paired_guide = partition_function_max_paired_guide[1]

    if str(subopt_structures_target[0].structure)[0:len_spacer] == '.' * len_spacer:
        ensemble_energy_max_paired_target = 0
    else:
        max_start_index_target, max_length_target = find_max_base_pairs(str(subopt_structures_target[0].structure)[0:len_spacer])
        max_paired_seq_target = guide[max_start_index_target:max_start_index_target+max_length_target]
        DNA_max_paired_seq_target = RNA_to_DNA(max_paired_seq_target)
        partition_function_max_paired_target = pfunc(strands=[DNA_max_paired_seq_target, DNA_reverse_complement(DNA_max_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_max_paired_target = partition_function_max_paired_target[1]

    # Calculate/compare ensemble_energy_max_unpaired_guide_spacer and ensemble_energy_max_unpaired_target_spacer
    if is_all_parens(str(subopt_structures_guide[0].structure)[0:len_spacer]) == True:
        ensemble_energy_max_unpaired_guide = 0
    else:
        max_start_index_guide, max_length_guide = find_max_unpaired(str(subopt_structures_guide[0].structure)[0:len_spacer])
        max_unpaired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_max_unpaired_guide = pfunc(strands=[max_unpaired_seq_guide, RNA_reverse_complement(max_unpaired_seq_guide)], model=my_model_RNA)
        ensemble_energy_max_unpaired_guide = partition_function_max_unpaired_guide[1]

    if is_all_parens(str(subopt_structures_target[0].structure)[0:len_spacer]) == True:
        ensemble_energy_max_unpaired_target = 0
    else:
        max_start_index_target, max_length_target = find_max_unpaired(str(subopt_structures_target[0].structure)[0:len_spacer])
        max_unpaired_seq_target = guide[max_start_index_target:max_start_index_target+max_length_target]
        DNA_max_unpaired_seq_target = RNA_to_DNA(max_unpaired_seq_target)
        partition_function_max_unpaired_target = pfunc(strands=[DNA_max_unpaired_seq_target, DNA_reverse_complement(DNA_max_unpaired_seq_target)], model=my_model_DNA)
        ensemble_energy_max_unpaired_target = partition_function_max_unpaired_target[1]

    # Calculate/compare ensemble_energy_5_overhang_guide_spacer and ensemble_energy_5_overhang_target_spacer
    if str(subopt_structures_guide[0].structure)[0] != '.':
        ensemble_energy_5_overhang_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_5_overhang(str(subopt_structures_guide[0].structure)[0:len_spacer])
        max_5_overhang_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_5_overhang_guide = pfunc(strands=[max_5_overhang_seq_guide, RNA_reverse_complement(max_5_overhang_seq_guide)], model=my_model_RNA)
        ensemble_energy_5_overhang_guide = partition_function_5_overhang_guide[1]

    if str(subopt_structures_target[0].structure)[0] != '.':
        ensemble_energy_5_overhang_target = 0
    else:
        max_start_index_target, max_length_target = detect_5_overhang(str(subopt_structures_target[0].structure)[0:len_spacer])
        max_5_overhang_seq_target = guide[max_start_index_target:max_start_index_target+max_length_target]
        DNA_5_overhang_seq_target = RNA_to_DNA(max_5_overhang_seq_target)
        partition_function_5_overhang_target = pfunc(strands=[DNA_5_overhang_seq_target, DNA_reverse_complement(DNA_5_overhang_seq_target)], model=my_model_DNA)
        ensemble_energy_5_overhang_target = partition_function_5_overhang_target[1]

    # Calculate/compare ensemble_energy_3_overhang_guide_spacer and ensemble_energy_3_overhang_target_spacer
    if str(subopt_structures_guide[0].structure)[len_spacer-1] != '.':
        ensemble_energy_3_overhang_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_3_overhang(str(subopt_structures_guide[0].structure)[0:len_spacer])
        max_3_overhang_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_3_overhang_guide = pfunc(strands=[max_3_overhang_seq_guide, RNA_reverse_complement(max_3_overhang_seq_guide)], model=my_model_RNA)
        ensemble_energy_3_overhang_guide = partition_function_3_overhang_guide[1]

    if str(subopt_structures_target[0].structure)[len_spacer-1] != '.':
        ensemble_energy_3_overhang_target = 0
    else:
        max_start_index_target, max_length_target = detect_3_overhang(str(subopt_structures_target[0].structure)[0:len_spacer])
        max_3_overhang_seq_target = guide[max_start_index_target:max_start_index_target+max_length_target]
        DNA_3_overhang_seq_target = RNA_to_DNA(max_3_overhang_seq_target)
        partition_function_3_overhang_target = pfunc(strands=[DNA_3_overhang_seq_target, DNA_reverse_complement(DNA_3_overhang_seq_target)], model=my_model_DNA)
        ensemble_energy_3_overhang_target = partition_function_3_overhang_target[1]

    # Calculate/compare ensemble_energy_5_paired_guide_spacer and ensemble_energy_5_paired_target_spacer
    if str(subopt_structures_guide[0].structure)[0] == '.':
        ensemble_energy_5_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_5_paired(str(subopt_structures_guide[0].structure)[0:len_spacer])
        max_5_paired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_5_paired_guide = pfunc(strands=[max_5_paired_seq_guide, RNA_reverse_complement(max_5_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_5_paired_guide = partition_function_5_paired_guide[1]

    if str(subopt_structures_target[0].structure)[0] == '.':
        ensemble_energy_5_paired_target = 0
    else:
        max_start_index_target, max_length_target = detect_5_paired(str(subopt_structures_target[0].structure)[0:len_spacer])
        max_5_paired_seq_target = guide[max_start_index_target:max_start_index_target+max_length_target]
        DNA_5_paired_seq_target = RNA_to_DNA(max_5_paired_seq_target)
        partition_function_5_paired_target = pfunc(strands=[DNA_5_paired_seq_target, DNA_reverse_complement(DNA_5_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_5_paired_target = partition_function_5_paired_target[1]
    
    # Calculate/compare ensemble_energy_3_paired_guide_spacer and ensemble_energy_3_paired_target_spacer
    if str(subopt_structures_guide[0].structure)[len_spacer-1] == '.':
        ensemble_energy_3_paired_guide = 0
    else:
        max_start_index_guide, max_length_guide = detect_3_paired(str(subopt_structures_guide[0].structure)[0:len_spacer])
        max_3_paired_seq_guide = guide[max_start_index_guide:max_start_index_guide+max_length_guide]
        partition_function_3_paired_guide = pfunc(strands=[max_3_paired_seq_guide, RNA_reverse_complement(max_3_paired_seq_guide)], model=my_model_RNA)
        ensemble_energy_3_paired_guide = partition_function_3_paired_guide[1]

    if str(subopt_structures_target[0].structure)[len_spacer-1] == '.':
        ensemble_energy_3_paired_target = 0
    else:
        max_start_index_target, max_length_target = detect_3_paired(str(subopt_structures_target[0].structure)[0:len_spacer])
        max_3_paired_seq_target = guide[max_start_index_target:max_start_index_target+max_length_target]
        DNA_3_paired_seq_target = RNA_to_DNA(max_3_paired_seq_target)
        partition_function_3_paired_target = pfunc(strands=[DNA_3_paired_seq_target, DNA_reverse_complement(DNA_3_paired_seq_target)], model=my_model_DNA)
        ensemble_energy_3_paired_target = partition_function_3_paired_target[1]

    #------Energy properties for the extension region of guide before and after hybridizing to its target (before cleavage)------
    len_PB_bc = len_spacer - get_initial_num_dots(triplex_bc_2)
    
    # Calculate/compare ensemble_energy_max_paired_guide_extension_bc and ensemble_energy_max_paired_target_extension_bc
    if str(subopt_structures_guide[0].structure)[-len_PB_bc:] == '.' * len_PB_bc:
        ensemble_energy_max_paired_guide_extension_bc = 0
    else:
        max_start_index_guide_extension, max_length_guide_extension = find_max_base_pairs(str(subopt_structures_guide[0].structure)[-len_PB_bc:])
        max_paired_seq_guide_extension = guide[max_start_index_guide_extension:max_start_index_guide_extension+max_length_guide_extension]
        partition_function_max_paired_guide_extension = pfunc(strands=[max_paired_seq_guide_extension, RNA_reverse_complement(max_paired_seq_guide_extension)], model=my_model_RNA)
        ensemble_energy_max_paired_guide_extension_bc = partition_function_max_paired_guide_extension[1]

    if triplex_bc_1[-len_PB_bc:] == '.' * len_PB_bc:
        ensemble_energy_max_paired_target_extension_bc = 0
    else:
        max_start_index_target_extension, max_length_target_extension = find_max_base_pairs(triplex_bc_1[-len_PB_bc:])
        max_paired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_max_paired_seq_target_extension = RNA_to_DNA(max_paired_seq_target_extension)
        partition_function_max_paired_target_extension = pfunc(strands=[DNA_max_paired_seq_target_extension, DNA_reverse_complement(DNA_max_paired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_max_paired_target_extension_bc = partition_function_max_paired_target_extension[1]

    # Calculate/compare ensemble_energy_max_unpaired_guide_extension_bc and ensemble_energy_max_unpaired_target_extension
    if is_all_parens(str(subopt_structures_guide[0].structure)[-len_PB_bc:]) == True:
        ensemble_energy_max_unpaired_guide_extension_bc = 0
    else:
        max_start_index_guide_extension, max_length_guide_extension = find_max_unpaired(str(subopt_structures_guide[0].structure)[-len_PB_bc:])
        max_unpaired_seq_guide_extension = guide[max_start_index_guide_extension:max_start_index_guide_extension+max_length_guide_extension]
        partition_function_max_unpaired_guide_extension = pfunc(strands=[max_unpaired_seq_guide_extension, RNA_reverse_complement(max_unpaired_seq_guide_extension)], model=my_model_RNA)
        ensemble_energy_max_unpaired_guide_extension_bc = partition_function_max_unpaired_guide_extension[1]

    if is_all_parens(triplex_bc_1[-len_PB_bc:]) == True:
        ensemble_energy_max_unpaired_target_extension_bc = 0
    else:
        max_start_index_target_extension, max_length_target_extension = find_max_unpaired(triplex_bc_1[-len_PB_bc:])
        max_unpaired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_max_unpaired_seq_target_extension = RNA_to_DNA(max_unpaired_seq_target_extension)
        partition_function_max_unpaired_target_extension = pfunc(strands=[DNA_max_unpaired_seq_target_extension, DNA_reverse_complement(DNA_max_unpaired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_max_unpaired_target_extension_bc = partition_function_max_unpaired_target_extension[1]
    
    # Calculate/compare ensemble_energy_5_overhang_guide_extension_bc and ensemble_energy_5_overhang_target_extension
    if str(subopt_structures_guide[0].structure)[-len_PB_bc] != '.':
        ensemble_energy_5_overhang_guide_extension_bc = 0
    else:
        max_start_index_guide_extension, max_length_guide_extension = detect_5_overhang(str(subopt_structures_guide[0].structure)[-len_PB_bc:])
        max_5_overhang_seq_guide_extension = guide[max_start_index_guide_extension:max_start_index_guide_extension+max_length_guide_extension]
        partition_function_5_overhang_guide_extension = pfunc(strands=[max_5_overhang_seq_guide_extension, RNA_reverse_complement(max_5_overhang_seq_guide_extension)], model=my_model_RNA)
        ensemble_energy_5_overhang_guide_extension_bc = partition_function_5_overhang_guide_extension[1]

    if triplex_bc_1[-len_PB_bc] != '.':
        ensemble_energy_5_overhang_target_extension_bc = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_5_overhang(triplex_bc_1[-len_PB_bc:])
        max_5_overhang_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_5_overhang_seq_target_extension = RNA_to_DNA(max_5_overhang_seq_target_extension)
        partition_function_5_overhang_target_extension = pfunc(strands=[DNA_5_overhang_seq_target_extension, DNA_reverse_complement(DNA_5_overhang_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_5_overhang_target_extension_bc = partition_function_5_overhang_target_extension[1]
    
    # Calculate/compare ensemble_energy_3_overhang_guide_extension_bc and ensemble_energy_3_overhang_target_extension
    if str(subopt_structures_guide[0].structure)[-1] != '.':
        ensemble_energy_3_overhang_guide_extension_bc = 0
    else:
        max_start_index_guide_extension, max_length_guide_extension = detect_3_overhang(str(subopt_structures_guide[0].structure)[-len_PB_bc:])
        max_3_overhang_seq_guide_extension = guide[max_start_index_guide_extension:max_start_index_guide_extension+max_length_guide_extension]
        partition_function_3_overhang_guide_extension = pfunc(strands=[max_3_overhang_seq_guide_extension, RNA_reverse_complement(max_3_overhang_seq_guide_extension)], model=my_model_RNA)
        ensemble_energy_3_overhang_guide_extension_bc = partition_function_3_overhang_guide_extension[1]

    if triplex_bc_1[-1] != '.':
        ensemble_energy_3_overhang_target_extension_bc = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_3_overhang(triplex_bc_1[-len_PB_bc:])
        max_3_overhang_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_3_overhang_seq_target_extension = RNA_to_DNA(max_3_overhang_seq_target_extension)
        partition_function_3_overhang_target_extension = pfunc(strands=[DNA_3_overhang_seq_target_extension, DNA_reverse_complement(DNA_3_overhang_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_3_overhang_target_extension_bc = partition_function_3_overhang_target_extension[1]
    
    # Calculate/compare ensemble_energy_5_paired_guide_extension_bc and ensemble_energy_5_paired_target_extension
    if str(subopt_structures_guide[0].structure)[-len_PB_bc] == '.':
        ensemble_energy_5_paired_guide_extension_bc = 0
    else:
        max_start_index_guide_extension, max_length_guide_extension = detect_5_paired(str(subopt_structures_guide[0].structure)[-len_PB_bc:])
        max_5_paired_seq_guide_extension = guide[max_start_index_guide_extension:max_start_index_guide_extension+max_length_guide_extension]
        partition_function_5_paired_guide_extension = pfunc(strands=[max_5_paired_seq_guide_extension, RNA_reverse_complement(max_5_paired_seq_guide_extension)], model=my_model_RNA)
        ensemble_energy_5_paired_guide_extension_bc = partition_function_5_paired_guide_extension[1]

    if triplex_bc_1[-len_PB_bc] == '.':
        ensemble_energy_5_paired_target_extension_bc = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_5_paired(triplex_bc_1[-len_PB_bc:])
        max_5_paired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_5_paired_seq_target_extension = RNA_to_DNA(max_5_paired_seq_target_extension)
        partition_function_5_paired_target_extension = pfunc(strands=[DNA_5_paired_seq_target_extension, DNA_reverse_complement(DNA_5_paired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_5_paired_target_extension_bc = partition_function_5_paired_target_extension[1]
    
    # Calculate/compare ensemble_energy_3_paired_guide_extension_bc and ensemble_energy_3_paired_target_extension
    if str(subopt_structures_guide[0].structure)[-1] == '.':
        ensemble_energy_3_paired_guide_extension_bc = 0
    else:
        max_start_index_guide_extension, max_length_guide_extension = detect_3_paired(str(subopt_structures_guide[0].structure)[-len_PB_bc:])
        max_3_paired_seq_guide_extension = guide[max_start_index_guide_extension:max_start_index_guide_extension+max_length_guide_extension]
        partition_function_3_paired_guide_extension = pfunc(strands=[max_3_paired_seq_guide_extension, RNA_reverse_complement(max_3_paired_seq_guide_extension)], model=my_model_RNA)
        ensemble_energy_3_paired_guide_extension_bc = partition_function_3_paired_guide_extension[1]

    if triplex_bc_1[-1] == '.':
        ensemble_energy_3_paired_target_extension_bc = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_3_paired(triplex_bc_1[-len_PB_bc:])
        max_3_paired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_3_paired_seq_target_extension = RNA_to_DNA(max_3_paired_seq_target_extension)
        partition_function_3_paired_target_extension = pfunc(strands=[DNA_3_paired_seq_target_extension, DNA_reverse_complement(DNA_3_paired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_3_paired_target_extension_bc = partition_function_3_paired_target_extension[1]
    
    #------Energy properties for the PBS region of guide after displaced target strand cleavage------
    len_PB_ac = len_spacer - get_initial_num_dots(triplex_ac_2)
    
    # Calculate/compare ensemble_energy_max_paired_target_PBS after cleavage
    if triplex_ac_1[-len_PB_ac:] == '.' * len_PB_ac:
        ensemble_energy_max_paired_target_extension_ac = 0
    else:
        max_start_index_target_extension, max_length_target_extension = find_max_base_pairs(triplex_ac_1[-len_PB_ac:])
        max_paired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_max_paired_seq_target_extension = RNA_to_DNA(max_paired_seq_target_extension)
        partition_function_max_paired_target_extension = pfunc(strands=[DNA_max_paired_seq_target_extension, DNA_reverse_complement(DNA_max_paired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_max_paired_target_extension_ac = partition_function_max_paired_target_extension[1]
    
    # Calculate/compare ensemble_energy_max_unpaired_target_PBS after cleavage
    if is_all_parens(triplex_ac_1[-len_PB_ac:]) == True:
        ensemble_energy_max_unpaired_target_extension_ac = 0
    else:
        max_start_index_target_extension, max_length_target_extension = find_max_unpaired(triplex_ac_1[-len_PB_ac:])
        max_unpaired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_max_unpaired_seq_target_extension = RNA_to_DNA(max_unpaired_seq_target_extension)
        partition_function_max_unpaired_target_extension = pfunc(strands=[DNA_max_unpaired_seq_target_extension, DNA_reverse_complement(DNA_max_unpaired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_max_unpaired_target_extension_ac = partition_function_max_unpaired_target_extension[1]
    
    # Calculate/compare ensemble_energy_5_overhang_target_PBS after cleavage
    if triplex_ac_1[-len_PB_ac] != '.':
        ensemble_energy_5_overhang_target_extension_ac = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_5_overhang(triplex_ac_1[-len_PB_ac:])
        max_5_overhang_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_5_overhang_seq_target_extension = RNA_to_DNA(max_5_overhang_seq_target_extension)
        partition_function_5_overhang_target_extension = pfunc(strands=[DNA_5_overhang_seq_target_extension, DNA_reverse_complement(DNA_5_overhang_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_5_overhang_target_extension_ac = partition_function_5_overhang_target_extension[1]
    
    # Calculate/compare ensemble_energy_3_overhang_target_PBS after cleavage
    if triplex_ac_1[-1] != '.':
        ensemble_energy_3_overhang_target_extension_ac = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_3_overhang(triplex_ac_1[-len_PB_ac:])
        max_3_overhang_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_3_overhang_seq_target_extension = RNA_to_DNA(max_3_overhang_seq_target_extension)
        partition_function_3_overhang_target_extension = pfunc(strands=[DNA_3_overhang_seq_target_extension, DNA_reverse_complement(DNA_3_overhang_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_3_overhang_target_extension_ac = partition_function_3_overhang_target_extension[1]
    
    # Calculate/compare ensemble_energy_5_paired_target_PBS after cleavage
    if triplex_ac_1[-len_PB_ac] == '.':
        ensemble_energy_5_paired_target_extension_ac = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_5_paired(triplex_ac_1[-len_PB_ac:])
        max_5_paired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_5_paired_seq_target_extension = RNA_to_DNA(max_5_paired_seq_target_extension)
        partition_function_5_paired_target_extension = pfunc(strands=[DNA_5_paired_seq_target_extension, DNA_reverse_complement(DNA_5_paired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_5_paired_target_extension_ac = partition_function_5_paired_target_extension[1]
    
    # Calculate/compare ensemble_energy_3_paired_target_PBS after cleavage
    if triplex_ac_1[-1] == '.':
        ensemble_energy_3_paired_target_extension_ac = 0
    else:
        max_start_index_target_extension, max_length_target_extension = detect_3_paired(triplex_ac_1[-len_PB_ac:])
        max_3_paired_seq_target_extension = guide[max_start_index_target_extension:max_start_index_target_extension+max_length_target_extension]
        DNA_3_paired_seq_target_extension = RNA_to_DNA(max_3_paired_seq_target_extension)
        partition_function_3_paired_target_extension = pfunc(strands=[DNA_3_paired_seq_target_extension, DNA_reverse_complement(DNA_3_paired_seq_target_extension)], model=my_model_DNA)
        ensemble_energy_3_paired_target_extension_ac = partition_function_3_paired_target_extension[1]
    
    #------Energy properties for the RT region of guide after displaced target strand cleavage------
    len_RT_ac = len(guide) - len_spacer - len(scaffold) - get_initial_num_dots(triplex_ac_2)
    
    # Calculate/compare ensemble_energy_max_paired_target_RT after cleavage
    if triplex_ac_1[100:100+len_RT_ac] == '.' * len_RT_ac:
        ensemble_energy_max_paired_target_RT = 0
    else:
        max_start_index_target_RT, max_length_target_RT = find_max_base_pairs(triplex_ac_1[100:100+len_RT_ac])
        max_paired_seq_target_RT = guide[max_start_index_target_RT:max_start_index_target_RT+max_length_target_RT]
        DNA_max_paired_seq_target_RT = RNA_to_DNA(max_paired_seq_target_RT)
        partition_function_max_paired_target_RT = pfunc(strands=[DNA_max_paired_seq_target_RT, DNA_reverse_complement(DNA_max_paired_seq_target_RT)], model=my_model_DNA)
        ensemble_energy_max_paired_target_RT = partition_function_max_paired_target_RT[1]
    
    # Calculate/compare ensemble_energy_max_unpaired_target_RT after cleavage
    if is_all_parens(triplex_ac_1[100:100+len_RT_ac]) == True:
        ensemble_energy_max_unpaired_target_RT = 0
    else:
        max_start_index_target_RT, max_length_target_RT = find_max_unpaired(triplex_ac_1[100:100+len_RT_ac])
        max_unpaired_seq_target_RT = guide[max_start_index_target_RT:max_start_index_target_RT+max_length_target_RT]
        DNA_max_unpaired_seq_target_RT = RNA_to_DNA(max_unpaired_seq_target_RT)
        partition_function_max_unpaired_target_RT = pfunc(strands=[DNA_max_unpaired_seq_target_RT, DNA_reverse_complement(DNA_max_unpaired_seq_target_RT)], model=my_model_DNA)
        ensemble_energy_max_unpaired_target_RT = partition_function_max_unpaired_target_RT[1]
    
    # Calculate/compare ensemble_energy_5_overhang_target_RT after cleavage
    if triplex_ac_1[100] != '.':
        ensemble_energy_5_overhang_target_RT = 0
    else:
        max_start_index_target_RT, max_length_target_RT = detect_5_overhang(triplex_ac_1[100:100+len_RT_ac])
        max_5_overhang_seq_target_RT = guide[max_start_index_target_RT:max_start_index_target_RT+max_length_target_RT]
        DNA_5_overhang_seq_target_RT = RNA_to_DNA(max_5_overhang_seq_target_RT)
        partition_function_5_overhang_target_RT = pfunc(strands=[DNA_5_overhang_seq_target_RT, DNA_reverse_complement(DNA_5_overhang_seq_target_RT)], model=my_model_DNA)
        ensemble_energy_5_overhang_target_RT = partition_function_5_overhang_target_RT[1]
    
     # Calculate/compare ensemble_energy_3_overhang_target_RT after cleavage
    if triplex_ac_1[100+len_RT_ac-1] != '.':
        ensemble_energy_3_overhang_target_RT = 0
    else:
        max_start_index_target_RT, max_length_target_RT = detect_3_overhang(triplex_ac_1[100:100+len_RT_ac])
        max_3_overhang_seq_target_RT = guide[max_start_index_target_RT:max_start_index_target_RT+max_length_target_RT]
        DNA_3_overhang_seq_target_RT = RNA_to_DNA(max_3_overhang_seq_target_RT)
        partition_function_3_overhang_target_RT = pfunc(strands=[DNA_3_overhang_seq_target_RT, DNA_reverse_complement(DNA_3_overhang_seq_target_RT)], model=my_model_DNA)
        ensemble_energy_3_overhang_target_RT = partition_function_3_overhang_target_RT[1]
    
    # Calculate/compare ensemble_energy_5_paired_target_RT after cleavage
    if triplex_ac_1[100] == '.':
        ensemble_energy_5_paired_target_RT = 0
    else:
        max_start_index_target_RT, max_length_target_RT = detect_5_paired(triplex_ac_1[100:100+len_RT_ac])
        max_5_paired_seq_target_RT = guide[max_start_index_target_RT:max_start_index_target_RT+max_length_target_RT]
        DNA_5_paired_seq_target_RT = RNA_to_DNA(max_5_paired_seq_target_RT)
        partition_function_5_paired_target_RT = pfunc(strands=[DNA_5_paired_seq_target_RT, DNA_reverse_complement(DNA_5_paired_seq_target_RT)], model=my_model_DNA)
        ensemble_energy_5_paired_target_RT = partition_function_5_paired_target_RT[1]
    
    # Calculate/compare ensemble_energy_3_paired_target_RT after cleavage
    if triplex_ac_1[100+len_RT_ac-1] == '.':
        ensemble_energy_3_paired_target_RT = 0
    else:
        max_start_index_target_RT, max_length_target_RT = detect_3_paired(triplex_ac_1[100:100+len_RT_ac])
        max_3_paired_seq_target_RT = guide[max_start_index_target_RT:max_start_index_target_RT+max_length_target_RT]
        DNA_3_paired_seq_target_RT = RNA_to_DNA(max_3_paired_seq_target_RT)
        partition_function_3_paired_target_RT = pfunc(strands=[DNA_3_paired_seq_target_RT, DNA_reverse_complement(DNA_3_paired_seq_target_RT)], model=my_model_DNA)
        ensemble_energy_3_paired_target_RT = partition_function_3_paired_target_RT[1]

    # Calculate/compare target seed region free energy 
    seed_region = target[0:6]
    subopt_structures_seed = subopt(strands=[seed_region, DNA_reverse_complement(seed_region)], energy_gap=3, model=my_model_DNA)
    seed_energy = subopt_structures_seed[0].energy

    # Calculate/compare target middle region free energy 
    middle_region = target[6:13]
    subopt_structures_middle = subopt(strands=[middle_region, DNA_reverse_complement(middle_region)], energy_gap=3, model=my_model_DNA)
    middle_energy = subopt_structures_middle[0].energy

    # Calculate/compare target distal region free energy 
    distal_region = target[13:20]
    subopt_structures_distal = subopt(strands=[distal_region, DNA_reverse_complement(distal_region)], energy_gap=3, model=my_model_DNA)
    distal_energy = subopt_structures_distal[0].energy

    energy_array_unit = [normalize_guide(subopt_structures_guide[0].energy - dGstruc_guide_linear)]
    energy_array_unit.append(normalize_guide_conse(ensemble_energy_max_paired_guide))
    energy_array_unit.append(normalize_guide_conse_ext_bc(ensemble_energy_max_paired_guide_extension_bc))
    energy_array_unit.append(normalize_guide_conse_unpaired(ensemble_energy_max_unpaired_guide))
    energy_array_unit.append(normalize_guide_conse_unpaired_ext_bc(ensemble_energy_max_unpaired_guide_extension_bc))
    energy_array_unit.append(normalize_guide_overhang(ensemble_energy_5_overhang_guide))
    energy_array_unit.append(normalize_guide_overhang_ext_bc(ensemble_energy_5_overhang_guide_extension_bc))
    energy_array_unit.append(normalize_guide_overhang(ensemble_energy_3_overhang_guide))
    energy_array_unit.append(normalize_guide_overhang_ext_bc(ensemble_energy_3_overhang_guide_extension_bc))
    energy_array_unit.append(normalize_guide_paired(ensemble_energy_5_paired_guide))
    energy_array_unit.append(normalize_guide_paired_ext_bc(ensemble_energy_5_paired_guide_extension_bc))
    energy_array_unit.append(normalize_guide_paired(ensemble_energy_3_paired_guide))
    energy_array_unit.append(normalize_guide_paired_ext_bc(ensemble_energy_3_paired_guide_extension_bc))
    
    energy_array_unit.append(normalize_target(subopt_structures_target[0].energy))
    energy_array_unit.append(normalize_target_conse(ensemble_energy_max_paired_target))
    energy_array_unit.append(normalize_target_conse_ext_bc(ensemble_energy_max_paired_target_extension_bc))
    energy_array_unit.append(normalize_target_conse_ext_ac(ensemble_energy_max_paired_target_extension_ac))
    energy_array_unit.append(normalize_target_conse_unpaired(ensemble_energy_max_unpaired_target))
    energy_array_unit.append(normalize_target_conse_unpaired_ext_bc(ensemble_energy_max_unpaired_target_extension_bc))
    energy_array_unit.append(normalize_target_conse_unpaired_ext_ac(ensemble_energy_max_unpaired_target_extension_ac))
    energy_array_unit.append(normalize_target_overhang(ensemble_energy_5_overhang_target))

    energy_array_unit.append(normalize_target_overhang_ext_bc(ensemble_energy_5_overhang_target_extension_bc))
    energy_array_unit.append(normalize_target_overhang_ext_ac(ensemble_energy_5_overhang_target_extension_ac))
    energy_array_unit.append(normalize_target_overhang(ensemble_energy_3_overhang_target))
    energy_array_unit.append(normalize_target_overhang_ext_bc(ensemble_energy_3_overhang_target_extension_bc))
    energy_array_unit.append(normalize_target_overhang_ext_ac(ensemble_energy_3_overhang_target_extension_ac))
    energy_array_unit.append(normalize_target_paired(ensemble_energy_5_paired_target))
    energy_array_unit.append(normalize_target_paired_ext_bc(ensemble_energy_5_paired_target_extension_bc))
    energy_array_unit.append(normalize_target_paired_ext_ac(ensemble_energy_5_paired_target_extension_ac))
    energy_array_unit.append(normalize_target_paired(ensemble_energy_3_paired_target))
    energy_array_unit.append(normalize_target_paired_ext_bc(ensemble_energy_3_paired_target_extension_bc))
    energy_array_unit.append(normalize_target_paired_ext_ac(ensemble_energy_3_paired_target_extension_ac))

    energy_array_unit.append(normalize_target_conse_RT(ensemble_energy_max_paired_target_RT))
    energy_array_unit.append(normalize_target_conse_unpaired_RT(ensemble_energy_max_unpaired_target_RT))
    energy_array_unit.append(normalize_target_overhang_RT(ensemble_energy_5_overhang_target_RT))
    energy_array_unit.append(normalize_target_overhang_RT(ensemble_energy_3_overhang_target_RT))
    energy_array_unit.append(normalize_target_paired_RT(ensemble_energy_5_paired_target_RT))
    energy_array_unit.append(normalize_target_paired_RT(ensemble_energy_3_paired_target_RT))

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

def build_guide_graph(guide, template):
    guide = guide + scaffold + template
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
    target = target[23:43]
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

def build_duplex_graph(guide, target, template):
    guide = guide + scaffold + template
    target = target[23:43]
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
    target = target[23:43]
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

def build_triplex_bc_graph(guide, target, template):
    guide = guide + scaffold + template
    target = target[23:43]
    triplex_seq = DNA_to_RNA(target) + guide + DNA_reverse_complement(target)
    prob_matrix = pairs(strands=[DNA_to_RNA(target) + guide, DNA_reverse_complement(target)], model=my_model_DNA).to_array()
    node_features = encode_sequence(triplex_seq)
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

def build_triplex_ac_graph(guide, target, template):
    guide = guide + scaffold + template
    target = target[23:43]
    triplex_seq = DNA_to_RNA(target) + guide + DNA_reverse_complement(target)[0:17]
    prob_matrix = pairs(strands=[DNA_to_RNA(target) + guide, DNA_reverse_complement(target)[0:17]], model=my_model_DNA).to_array()
    node_features = encode_sequence(triplex_seq)
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
    