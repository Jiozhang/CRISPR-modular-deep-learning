from dataclasses import dataclass
from nupack import *
from torch import seed

def DNA_to_RNA(DNA):
    match = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'U'}
    return ''.join(match.get(base, base) for base in (DNA))

def DNA_reverse_complement(DNA):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(DNA))

def RNA_reverse_complement(RNA):
    complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(RNA))

# Define NUPACK model
my_model_RNA = Model(material='rna', ensemble='nostacking', celsius=37, sodium=0.1, magnesium=0.01)
my_model_DNA = Model(material='dna', ensemble='nostacking', celsius=37, sodium=0.1, magnesium=0.01)

@dataclass
class CustomFeatureParams:
    mode: str
    scaffold_seq: str
    spacer_len: int

    # normalization parameters
    guide_min: float
    guide_max: float

    target_min: float
    target_max: float

    ss_target_bh_min: float
    ss_target_bh_max: float

    guide_conse_min: float
    guide_conse_max: float

    target_conse_min: float
    target_conse_max: float

    ss_target_bh_conse_min: float
    ss_target_bh_conse_max: float

    guide_conse_unpaired_min: float
    guide_conse_unpaired_max: float

    target_conse_unpaired_min: float
    target_conse_unpaired_max: float

    ss_target_bh_conse_unpaired_min: float
    ss_target_bh_conse_unpaired_max: float

    guide_overhang_min: float
    guide_overhang_max: float

    target_overhang_min: float
    target_overhang_max: float

    ss_target_bh_overhang_min: float
    ss_target_bh_overhang_max: float
 
    guide_paired_min: float
    guide_paired_max: float

    target_paired_min: float
    target_paired_max: float

    ss_target_bh_paired_min: float
    ss_target_bh_paired_max: float

    seed_min: float
    seed_max: float

    middle_min: float
    middle_max: float

    distal_min: float
    distal_max: float

    target_conse_3_min: float
    target_conse_3_max: float

    target_conse_4_min: float
    target_conse_4_max: float

    target_conse_5_min: float
    target_conse_5_max: float

    target_conse_6_min: float
    target_conse_6_max: float

    target_conse_7_min: float
    target_conse_7_max: float

    target_conse_8_min: float
    target_conse_8_max: float

def clean_seq(seq: str) -> str:
    return str(seq).strip().upper().replace(" ", "").replace("\n", "")

def calculate_custom_feature_params(
    scaffold_seq: str,
    spacer_len: int,
    mode: str,
) -> CustomFeatureParams:
    scaffold_seq = clean_seq(scaffold_seq)

    if spacer_len <= 0:
        raise ValueError("Spacer length must be positive.")

    if mode not in {"Cas9", "Cas12", "Cas13"}:
        raise ValueError(f"Unsupported mode: {mode}")
    
    if spacer_len % 2 == 0:
        ref_seq_full_GC = 'GC' * (spacer_len//2)
        ref_seq_full_TA = 'TA' * (spacer_len//2)
        ref_seq_full_UA = 'UA' * (spacer_len//2)
    else:
        ref_seq_full_GC = 'GC' * (spacer_len//2) + 'G'
        ref_seq_full_TA = 'TA' * (spacer_len//2) + 'T'
        ref_seq_full_UA = 'UA' * (spacer_len//2) + 'U'
    
    ref_seq_single_A = 'A'
    ref_seq_conse_GC = 'GC' * (spacer_len//4)
    ref_seq_full_G = 'G' * spacer_len

    ref_seq_full_GC_seed = 'GC' * (spacer_len//3)
    ref_seq_full_TA_seed = 'TA' * (spacer_len//3)
    ref_seq_full_UA_seed = 'UA' * (spacer_len//3)

    if spacer_len % 3 == 0:
        ref_seq_full_GC_middle = 'GC' * (spacer_len//3)
        ref_seq_full_TA_middle = 'TA' * (spacer_len//3)
        ref_seq_full_UA_middle = 'UA' * (spacer_len//3)
        ref_seq_full_GC_distal = 'GC' * (spacer_len//3)
        ref_seq_full_TA_distal = 'TA' * (spacer_len//3)
        ref_seq_full_UA_distal = 'UA' * (spacer_len//3)
    elif spacer_len % 3 == 1:
        ref_seq_full_GC_middle = 'GC' * (spacer_len//3) + 'G'
        ref_seq_full_TA_middle = 'TA' * (spacer_len//3) + 'T'
        ref_seq_full_UA_middle = 'UA' * (spacer_len//3) + 'U'
        ref_seq_full_GC_distal = 'GC' * (spacer_len//3)
        ref_seq_full_TA_distal = 'TA' * (spacer_len//3)
        ref_seq_full_UA_distal = 'UA' * (spacer_len//3)
    elif spacer_len % 3 == 2:
        ref_seq_full_GC_middle = 'GC' * (spacer_len//3) + 'G'
        ref_seq_full_TA_middle = 'TA' * (spacer_len//3) + 'T'
        ref_seq_full_UA_middle = 'UA' * (spacer_len//3) + 'U'
        ref_seq_full_GC_distal = 'GC' * (spacer_len//3) + 'G'
        ref_seq_full_TA_distal = 'TA' * (spacer_len//3) + 'T'
        ref_seq_full_UA_distal = 'UA' * (spacer_len//3) + 'U'

    ref_seq_spacer_3GC = 'GCG'
    ref_seq_spacer_3UA = 'UAU'
    subopt_structures_spacer_3_GC = subopt(strands=[ref_seq_spacer_3GC, RNA_reverse_complement(ref_seq_spacer_3GC)], energy_gap=0.01, model=my_model_RNA)
    subopt_structures_spacer_3_UA = subopt(strands=[ref_seq_spacer_3UA, RNA_reverse_complement(ref_seq_spacer_3UA)], energy_gap=0.01, model=my_model_RNA)

    ref_seq_spacer_4GC = 'GCGC'
    ref_seq_spacer_4UA = 'UAUA'
    subopt_structures_spacer_4_GC = subopt(strands=[ref_seq_spacer_4GC, RNA_reverse_complement(ref_seq_spacer_4GC)], energy_gap=0.01, model=my_model_RNA)
    subopt_structures_spacer_4_UA = subopt(strands=[ref_seq_spacer_4UA, RNA_reverse_complement(ref_seq_spacer_4UA)], energy_gap=0.01, model=my_model_RNA)

    ref_seq_spacer_5GC = 'GCGCG'
    ref_seq_spacer_5UA = 'UAUAU'
    subopt_structures_spacer_5_GC = subopt(strands=[ref_seq_spacer_5GC, RNA_reverse_complement(ref_seq_spacer_5GC)], energy_gap=0.01, model=my_model_RNA)
    subopt_structures_spacer_5_UA = subopt(strands=[ref_seq_spacer_5UA, RNA_reverse_complement(ref_seq_spacer_5UA)], energy_gap=0.01, model=my_model_RNA)

    ref_seq_spacer_6GC = 'GCGCGC'
    ref_seq_spacer_6UA = 'UAUAUA'
    subopt_structures_spacer_6_GC = subopt(strands=[ref_seq_spacer_6GC, RNA_reverse_complement(ref_seq_spacer_6GC)], energy_gap=0.01, model=my_model_RNA)
    subopt_structures_spacer_6_UA = subopt(strands=[ref_seq_spacer_6UA, RNA_reverse_complement(ref_seq_spacer_6UA)], energy_gap=0.01, model=my_model_RNA)

    ref_seq_spacer_7GC = 'GCGCGCG'
    ref_seq_spacer_7UA = 'UAUAUAU'
    subopt_structures_spacer_7_GC = subopt(strands=[ref_seq_spacer_7GC, RNA_reverse_complement(ref_seq_spacer_7GC)], energy_gap=0.01, model=my_model_RNA)
    subopt_structures_spacer_7_UA = subopt(strands=[ref_seq_spacer_7UA, RNA_reverse_complement(ref_seq_spacer_7UA)], energy_gap=0.01, model=my_model_RNA)

    ref_seq_spacer_8GC = 'GCGCGCGC' 
    ref_seq_spacer_8UA = 'UAUAUAUA'
    subopt_structures_spacer_8_GC = subopt(strands=[ref_seq_spacer_8GC, RNA_reverse_complement(ref_seq_spacer_8GC)], energy_gap=0.01, model=my_model_RNA)
    subopt_structures_spacer_8_UA = subopt(strands=[ref_seq_spacer_8UA, RNA_reverse_complement(ref_seq_spacer_8UA)], energy_gap=0.01, model=my_model_RNA)
        
    subopt_structures_scaffold = subopt(strands=scaffold_seq, energy_gap=0.01, model=my_model_RNA)
    

    # Replace all of these with your real calculation functions
    if mode == "Cas9":
        full_guide_GC = DNA_to_RNA(ref_seq_full_GC) + scaffold_seq 
        full_guide_TA = DNA_to_RNA(ref_seq_full_TA) + scaffold_seq 

        # Calculate guide_min and guide_max
        subopt_structures_guide_GC = subopt(strands=full_guide_GC, energy_gap=0.01, model=my_model_RNA)
        guide_min = subopt_structures_guide_GC[0].energy - subopt_structures_scaffold[0].energy
        guide_max = subopt_structures_scaffold[0].energy - subopt_structures_scaffold[0].energy

        # Calculate target_min and target_max
        subopt_structures_duplex_GC = subopt(strands=[full_guide_GC, DNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_duplex_TA = subopt(strands=[full_guide_TA, DNA_reverse_complement(ref_seq_full_TA)], energy_gap=0.01, model=my_model_DNA)
        target_min = subopt_structures_duplex_GC[0].energy
        target_max = subopt_structures_duplex_TA[0].energy

        # Calculate ss_target_bh_min and ss_target_bh_max
        subopt_structures_ss_target_GC = subopt(strands=ref_seq_full_GC, energy_gap=0.01, model=my_model_DNA)
        ss_target_bh_min = subopt_structures_ss_target_GC[0].energy
        ss_target_bh_max = 0

        # Calculate guide_conse_min and guide_conse_max
        subopt_structures_ds_target_GC_RNA = subopt(strands=[ref_seq_full_GC, RNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_ds_target_single_A_RNA = subopt(strands=[ref_seq_single_A, RNA_reverse_complement(ref_seq_single_A)], energy_gap=0.01, model=my_model_RNA)
        guide_conse_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_conse_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_conse_min and target_conse_max
        subopt_structures_ds_target_GC_DNA = subopt(strands=[ref_seq_full_GC, DNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_ds_target_single_A_DNA = subopt(strands=[ref_seq_single_A, DNA_reverse_complement(ref_seq_single_A)], energy_gap=0.01, model=my_model_DNA)
        target_conse_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_conse_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_conse_min and ss_target_bh_conse_max
        subopt_structures_ref_seq_conse_GC = subopt(strands=[ref_seq_conse_GC, DNA_reverse_complement(ref_seq_conse_GC)], energy_gap=0.01, model=my_model_DNA)
        ss_target_bh_conse_min = subopt_structures_ref_seq_conse_GC[0].energy
        ss_target_bh_conse_max = 0

        # Calculate guide_conse_unpaired_min and guide_conse_unpaired_max
        guide_conse_unpaired_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_conse_unpaired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_conse_unpaired_min and target_conse_unpaired_max
        target_conse_unpaired_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_conse_unpaired_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_conse_unpaired_min and ss_target_bh_conse_unpaired_max
        subopt_structures_ref_seq_full_G = subopt(strands=[ref_seq_full_G, DNA_reverse_complement(ref_seq_full_G)], energy_gap=0.01, model=my_model_DNA)
        ss_target_bh_conse_unpaired_min = subopt_structures_ref_seq_full_G[0].energy
        ss_target_bh_conse_unpaired_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate guide_overhang_min and guide_overhang_max
        guide_overhang_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_overhang_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_overhang_min and target_overhang_max
        target_overhang_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_overhang_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_overhang_min and ss_target_bh_overhang_max
        ss_target_bh_overhang_min = subopt_structures_ref_seq_full_G[0].energy
        ss_target_bh_overhang_max = subopt_structures_ds_target_single_A_DNA[0].energy
     
        # Calculate guide_paired_min and guide_paired_max
        guide_paired_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_paired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_paired_min and target_paired_max
        target_paired_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_paired_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_paired_min and ss_target_bh_paired_max
        ss_target_bh_paired_min = subopt_structures_ref_seq_conse_GC[0].energy
        ss_target_bh_paired_max = 0

        # Calculate seed_min and seed_max
        subopt_structures_seed_GC = subopt(strands=[ref_seq_full_GC_seed, DNA_reverse_complement(ref_seq_full_GC_seed)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_seed_TA = subopt(strands=[ref_seq_full_TA_seed, DNA_reverse_complement(ref_seq_full_TA_seed)], energy_gap=0.01, model=my_model_DNA)
        seed_min = subopt_structures_seed_GC[0].energy
        seed_max = subopt_structures_seed_TA[0].energy

        # Calculate middle_min and middle_max
        subopt_structures_middle_GC = subopt(strands=[ref_seq_full_GC_middle, DNA_reverse_complement(ref_seq_full_GC_middle)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_middle_TA = subopt(strands=[ref_seq_full_TA_middle, DNA_reverse_complement(ref_seq_full_TA_middle)], energy_gap=0.01, model=my_model_DNA)
        middle_min = subopt_structures_middle_GC[0].energy
        middle_max = subopt_structures_middle_TA[0].energy

        # Calculate distal_min and distal_max
        subopt_structures_distal_GC = subopt(strands=[ref_seq_full_GC_distal, DNA_reverse_complement(ref_seq_full_GC_distal)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_distal_TA = subopt(strands=[ref_seq_full_TA_distal, DNA_reverse_complement(ref_seq_full_TA_distal)], energy_gap=0.01, model=my_model_DNA)
        distal_min = subopt_structures_distal_GC[0].energy
        distal_max = subopt_structures_distal_TA[0].energy

        # Calculate target_conse_3_min and target_conse_3_max
        target_conse_3_min = subopt_structures_spacer_3_GC[0].energy
        target_conse_3_max = subopt_structures_spacer_3_UA[0].energy

        # Calculate target_conse_4_min and target_conse_4_max
        target_conse_4_min = subopt_structures_spacer_4_GC[0].energy
        target_conse_4_max = subopt_structures_spacer_4_UA[0].energy

        # Calculate target_conse_5_min and target_conse_5_max
        target_conse_5_min = subopt_structures_spacer_5_GC[0].energy
        target_conse_5_max = subopt_structures_spacer_5_UA[0].energy

        # Calculate target_conse_6_min and target_conse_6_max
        target_conse_6_min = subopt_structures_spacer_6_GC[0].energy
        target_conse_6_max = subopt_structures_spacer_6_UA[0].energy

        # Calculate target_conse_7_min and target_conse_7_max
        target_conse_7_min = subopt_structures_spacer_7_GC[0].energy
        target_conse_7_max = subopt_structures_spacer_7_UA[0].energy

        # Calculate target_conse_8_min and target_conse_8_max
        target_conse_8_min = subopt_structures_spacer_8_GC[0].energy
        target_conse_8_max = subopt_structures_spacer_8_UA[0].energy

    elif mode == "Cas12":  # Cas12
        full_guide_GC = scaffold_seq + DNA_to_RNA(ref_seq_full_GC)
        full_guide_TA = scaffold_seq + DNA_to_RNA(ref_seq_full_TA)

        # Calculate guide_min and guide_max
        subopt_structures_guide_GC = subopt(strands=full_guide_GC, energy_gap=0.01, model=my_model_RNA)
        guide_min = subopt_structures_guide_GC[0].energy - subopt_structures_scaffold[0].energy
        guide_max = subopt_structures_scaffold[0].energy - subopt_structures_scaffold[0].energy

        # Calculate target_min and target_max
        subopt_structures_duplex_GC = subopt(strands=[full_guide_GC, DNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_duplex_TA = subopt(strands=[full_guide_TA, DNA_reverse_complement(ref_seq_full_TA)], energy_gap=0.01, model=my_model_DNA)
        target_min = subopt_structures_duplex_GC[0].energy
        target_max = subopt_structures_duplex_TA[0].energy

        # Calculate ss_target_bh_min and ss_target_bh_max
        subopt_structures_ss_target_GC = subopt(strands=ref_seq_full_GC, energy_gap=0.01, model=my_model_DNA)
        ss_target_bh_min = subopt_structures_ss_target_GC[0].energy
        ss_target_bh_max = 0

        # Calculate guide_conse_min and guide_conse_max
        subopt_structures_ds_target_GC_RNA = subopt(strands=[ref_seq_full_GC, RNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_ds_target_single_A_RNA = subopt(strands=[ref_seq_single_A, RNA_reverse_complement(ref_seq_single_A)], energy_gap=0.01, model=my_model_RNA)
        guide_conse_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_conse_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_conse_min and target_conse_max
        subopt_structures_ds_target_GC_DNA = subopt(strands=[ref_seq_full_GC, DNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_ds_target_single_A_DNA = subopt(strands=[ref_seq_single_A, DNA_reverse_complement(ref_seq_single_A)], energy_gap=0.01, model=my_model_DNA)
        target_conse_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_conse_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_conse_min and ss_target_bh_conse_max
        subopt_structures_ref_seq_conse_GC = subopt(strands=[ref_seq_conse_GC, DNA_reverse_complement(ref_seq_conse_GC)], energy_gap=0.01, model=my_model_DNA)
        ss_target_bh_conse_min = subopt_structures_ref_seq_conse_GC[0].energy
        ss_target_bh_conse_max = 0

        # Calculate guide_conse_unpaired_min and guide_conse_unpaired_max
        guide_conse_unpaired_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_conse_unpaired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_conse_unpaired_min and target_conse_unpaired_max
        target_conse_unpaired_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_conse_unpaired_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_conse_unpaired_min and ss_target_bh_conse_unpaired_max
        subopt_structures_ref_seq_full_G = subopt(strands=[ref_seq_full_G, DNA_reverse_complement(ref_seq_full_G)], energy_gap=0.01, model=my_model_DNA)
        ss_target_bh_conse_unpaired_min = subopt_structures_ref_seq_full_G[0].energy
        ss_target_bh_conse_unpaired_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate guide_overhang_min and guide_overhang_max
        guide_overhang_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_overhang_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_overhang_min and target_overhang_max
        target_overhang_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_overhang_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_overhang_min and ss_target_bh_overhang_max
        ss_target_bh_overhang_min = subopt_structures_ref_seq_full_G[0].energy
        ss_target_bh_overhang_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate guide_paired_min and guide_paired_max
        guide_paired_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_paired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_paired_min and target_paired_max
        target_paired_min = subopt_structures_ds_target_GC_DNA[0].energy
        target_paired_max = subopt_structures_ds_target_single_A_DNA[0].energy

        # Calculate ss_target_bh_paired_min and ss_target_bh_paired_max
        ss_target_bh_paired_min = subopt_structures_ref_seq_conse_GC[0].energy
        ss_target_bh_paired_max = 0

        # Calculate seed_min and seed_max
        subopt_structures_seed_GC = subopt(strands=[ref_seq_full_GC_seed, DNA_reverse_complement(ref_seq_full_GC_seed)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_seed_TA = subopt(strands=[ref_seq_full_TA_seed, DNA_reverse_complement(ref_seq_full_TA_seed)], energy_gap=0.01, model=my_model_DNA)
        seed_min = subopt_structures_seed_GC[0].energy
        seed_max = subopt_structures_seed_TA[0].energy

        # Calculate middle_min and middle_max
        subopt_structures_middle_GC = subopt(strands=[ref_seq_full_GC_middle, DNA_reverse_complement(ref_seq_full_GC_middle)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_middle_TA = subopt(strands=[ref_seq_full_TA_middle, DNA_reverse_complement(ref_seq_full_TA_middle)], energy_gap=0.01, model=my_model_DNA)
        middle_min = subopt_structures_middle_GC[0].energy
        middle_max = subopt_structures_middle_TA[0].energy

        # Calculate distal_min and distal_max
        subopt_structures_distal_GC = subopt(strands=[ref_seq_full_GC_distal, DNA_reverse_complement(ref_seq_full_GC_distal)], energy_gap=0.01, model=my_model_DNA)
        subopt_structures_distal_TA = subopt(strands=[ref_seq_full_TA_distal, DNA_reverse_complement(ref_seq_full_TA_distal)], energy_gap=0.01, model=my_model_DNA)
        distal_min = subopt_structures_distal_GC[0].energy
        distal_max = subopt_structures_distal_TA[0].energy

        # Calculate target_conse_3_min and target_conse_3_max
        target_conse_3_min = subopt_structures_spacer_3_GC[0].energy
        target_conse_3_max = subopt_structures_spacer_3_UA[0].energy

        # Calculate target_conse_4_min and target_conse_4_max
        target_conse_4_min = subopt_structures_spacer_4_GC[0].energy
        target_conse_4_max = subopt_structures_spacer_4_UA[0].energy

        # Calculate target_conse_5_min and target_conse_5_max
        target_conse_5_min = subopt_structures_spacer_5_GC[0].energy
        target_conse_5_max = subopt_structures_spacer_5_UA[0].energy

        # Calculate target_conse_6_min and target_conse_6_max
        target_conse_6_min = subopt_structures_spacer_6_GC[0].energy
        target_conse_6_max = subopt_structures_spacer_6_UA[0].energy

        # Calculate target_conse_7_min and target_conse_7_max
        target_conse_7_min = subopt_structures_spacer_7_GC[0].energy
        target_conse_7_max = subopt_structures_spacer_7_UA[0].energy

        # Calculate target_conse_8_min and target_conse_8_max
        target_conse_8_min = subopt_structures_spacer_8_GC[0].energy
        target_conse_8_max = subopt_structures_spacer_8_UA[0].energy
    
    elif mode == "Cas13":  # Cas13
        full_guide_GC = scaffold_seq + DNA_to_RNA(ref_seq_full_GC)
        full_guide_UA = scaffold_seq + DNA_to_RNA(ref_seq_full_UA)

        # Calculate guide_min and guide_max
        subopt_structures_guide_GC = subopt(strands=full_guide_GC, energy_gap=0.01, model=my_model_RNA)
        guide_min = subopt_structures_guide_GC[0].energy - subopt_structures_scaffold[0].energy
        guide_max = subopt_structures_scaffold[0].energy - subopt_structures_scaffold[0].energy

        # Calculate target_min and target_max
        subopt_structures_duplex_GC = subopt(strands=[full_guide_GC, RNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_duplex_UA = subopt(strands=[full_guide_UA, RNA_reverse_complement(ref_seq_full_UA)], energy_gap=0.01, model=my_model_RNA)
        target_min = subopt_structures_duplex_GC[0].energy
        target_max = subopt_structures_duplex_UA[0].energy

        # Calculate ss_target_bh_min and ss_target_bh_max
        subopt_structures_ss_target_GC = subopt(strands=ref_seq_full_GC, energy_gap=0.01, model=my_model_RNA)
        ss_target_bh_min = subopt_structures_ss_target_GC[0].energy
        ss_target_bh_max = 0

        # Calculate guide_conse_min and guide_conse_max
        subopt_structures_ds_target_GC_RNA = subopt(strands=[ref_seq_full_GC, RNA_reverse_complement(ref_seq_full_GC)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_ds_target_single_A_RNA = subopt(strands=[ref_seq_single_A, RNA_reverse_complement(ref_seq_single_A)], energy_gap=0.01, model=my_model_RNA)
        guide_conse_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_conse_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_conse_min and target_conse_max
        target_conse_min = subopt_structures_ds_target_GC_RNA[0].energy
        target_conse_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate ss_target_bh_conse_min and ss_target_bh_conse_max
        subopt_structures_ref_seq_conse_GC = subopt(strands=[ref_seq_conse_GC, RNA_reverse_complement(ref_seq_conse_GC)], energy_gap=0.01, model=my_model_RNA)
        ss_target_bh_conse_min = subopt_structures_ref_seq_conse_GC[0].energy
        ss_target_bh_conse_max = 0

        # Calculate guide_conse_unpaired_min and guide_conse_unpaired_max
        guide_conse_unpaired_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_conse_unpaired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_conse_unpaired_min and target_conse_unpaired_max
        target_conse_unpaired_min = subopt_structures_ds_target_GC_RNA[0].energy
        target_conse_unpaired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate ss_target_bh_conse_unpaired_min and ss_target_bh_conse_unpaired_max
        subopt_structures_ref_seq_full_G = subopt(strands=[ref_seq_full_G, RNA_reverse_complement(ref_seq_full_G)], energy_gap=0.01, model=my_model_RNA)
        ss_target_bh_conse_unpaired_min = subopt_structures_ref_seq_full_G[0].energy
        ss_target_bh_conse_unpaired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate guide_overhang_min and guide_overhang_max
        guide_overhang_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_overhang_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_overhang_min and target_overhang_max
        target_overhang_min = subopt_structures_ds_target_GC_RNA[0].energy
        target_overhang_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate ss_target_bh_overhang_min and ss_target_bh_overhang_max
        ss_target_bh_overhang_min = subopt_structures_ref_seq_full_G[0].energy
        ss_target_bh_overhang_max = subopt_structures_ds_target_single_A_RNA[0].energy
        
        # Calculate guide_paired_min and guide_paired_max
        guide_paired_min = subopt_structures_ds_target_GC_RNA[0].energy
        guide_paired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate target_paired_min and target_paired_max
        target_paired_min = subopt_structures_ds_target_GC_RNA[0].energy
        target_paired_max = subopt_structures_ds_target_single_A_RNA[0].energy

        # Calculate ss_target_bh_paired_min and ss_target_bh_paired_max
        ss_target_bh_paired_min = subopt_structures_ref_seq_conse_GC[0].energy
        ss_target_bh_paired_max = 0

        # Calculate seed_min and seed_max
        subopt_structures_seed_GC = subopt(strands=[ref_seq_full_GC_seed, DNA_reverse_complement(ref_seq_full_GC_seed)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_seed_UA = subopt(strands=[ref_seq_full_UA_seed, DNA_reverse_complement(ref_seq_full_UA_seed)], energy_gap=0.01, model=my_model_RNA)
        seed_min = subopt_structures_seed_GC[0].energy
        seed_max = subopt_structures_seed_UA[0].energy

        # Calculate middle_min and middle_max
        subopt_structures_middle_GC = subopt(strands=[ref_seq_full_GC_middle, DNA_reverse_complement(ref_seq_full_GC_middle)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_middle_UA = subopt(strands=[ref_seq_full_UA_middle, DNA_reverse_complement(ref_seq_full_UA_middle)], energy_gap=0.01, model=my_model_RNA)
        middle_min = subopt_structures_middle_GC[0].energy
        middle_max = subopt_structures_middle_UA[0].energy

        # Calculate distal_min and distal_max
        subopt_structures_distal_GC = subopt(strands=[ref_seq_full_GC_distal, DNA_reverse_complement(ref_seq_full_GC_distal)], energy_gap=0.01, model=my_model_RNA)
        subopt_structures_distal_UA = subopt(strands=[ref_seq_full_UA_distal, DNA_reverse_complement(ref_seq_full_UA_distal)], energy_gap=0.01, model=my_model_RNA)
        distal_min = subopt_structures_distal_GC[0].energy
        distal_max = subopt_structures_distal_UA[0].energy

        # Calculate target_conse_3_min and target_conse_3_max
        target_conse_3_min = subopt_structures_spacer_3_GC[0].energy
        target_conse_3_max = subopt_structures_spacer_3_UA[0].energy

        # Calculate target_conse_4_min and target_conse_4_max
        target_conse_4_min = subopt_structures_spacer_4_GC[0].energy
        target_conse_4_max = subopt_structures_spacer_4_UA[0].energy

        # Calculate target_conse_5_min and target_conse_5_max
        target_conse_5_min = subopt_structures_spacer_5_GC[0].energy
        target_conse_5_max = subopt_structures_spacer_5_UA[0].energy

        # Calculate target_conse_6_min and target_conse_6_max
        target_conse_6_min = subopt_structures_spacer_6_GC[0].energy
        target_conse_6_max = subopt_structures_spacer_6_UA[0].energy

        # Calculate target_conse_7_min and target_conse_7_max
        target_conse_7_min = subopt_structures_spacer_7_GC[0].energy
        target_conse_7_max = subopt_structures_spacer_7_UA[0].energy

        # Calculate target_conse_8_min and target_conse_8_max
        target_conse_8_min = subopt_structures_spacer_8_GC[0].energy
        target_conse_8_max = subopt_structures_spacer_8_UA[0].energy

    guide_total_len = len(scaffold_seq) + spacer_len

    return CustomFeatureParams(
        mode=mode,
        scaffold_seq=scaffold_seq,
        spacer_len=spacer_len,
        guide_min=guide_min,
        guide_max=guide_max,
        target_min=target_min,
        target_max=target_max,
        ss_target_bh_min=ss_target_bh_min,
        ss_target_bh_max=ss_target_bh_max,
        guide_conse_min=guide_conse_min,
        guide_conse_max=guide_conse_max,
        target_conse_min=target_conse_min,
        target_conse_max=target_conse_max,
        ss_target_bh_conse_min=ss_target_bh_conse_min,
        ss_target_bh_conse_max=ss_target_bh_conse_max,
        guide_conse_unpaired_min=guide_conse_unpaired_min,
        guide_conse_unpaired_max=guide_conse_unpaired_max,
        target_conse_unpaired_min=target_conse_unpaired_min,
        target_conse_unpaired_max=target_conse_unpaired_max,
        ss_target_bh_conse_unpaired_min=ss_target_bh_conse_unpaired_min,
        ss_target_bh_conse_unpaired_max=ss_target_bh_conse_unpaired_max,
        guide_overhang_min=guide_overhang_min,
        guide_overhang_max=guide_overhang_max,
        target_overhang_min=target_overhang_min,
        target_overhang_max=target_overhang_max,
        ss_target_bh_overhang_min=ss_target_bh_overhang_min,
        ss_target_bh_overhang_max=ss_target_bh_overhang_max,
        guide_paired_min=guide_paired_min,
        guide_paired_max=guide_paired_max,
        target_paired_min=target_paired_min,
        target_paired_max=target_paired_max,
        ss_target_bh_paired_min=ss_target_bh_paired_min,
        ss_target_bh_paired_max=ss_target_bh_paired_max,
        seed_min=seed_min,
        seed_max=seed_max,
        middle_min=middle_min,
        middle_max=middle_max,
        distal_min=distal_min,
        distal_max=distal_max,
        target_conse_3_min=target_conse_3_min,
        target_conse_3_max=target_conse_3_max,
        target_conse_4_min=target_conse_4_min,
        target_conse_4_max=target_conse_4_max,
        target_conse_5_min=target_conse_5_min,
        target_conse_5_max=target_conse_5_max,
        target_conse_6_min=target_conse_6_min,
        target_conse_6_max=target_conse_6_max,
        target_conse_7_min=target_conse_7_min,
        target_conse_7_max=target_conse_7_max,
        target_conse_8_min=target_conse_8_min,
        target_conse_8_max=target_conse_8_max,
    )