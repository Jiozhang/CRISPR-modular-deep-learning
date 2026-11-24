from dataclasses import dataclass
from ViennaRNA import RNA
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

# Define ViennaRNA model
temp = 37
salt = 0.1
RNA.cvar.temperature = temp
RNA.cvar.salt = salt

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
    RNA.params_load_RNA_Turner2004()
    fc_spacer_3_GC = RNA.fold_compound(f"{ref_seq_spacer_3GC}&{RNA_reverse_complement(ref_seq_spacer_3GC)}")
    energy_spacer_3_GC = fc_spacer_3_GC.mfe()[1]
    fc_spacer_3_UA = RNA.fold_compound(f"{ref_seq_spacer_3UA}&{RNA_reverse_complement(ref_seq_spacer_3UA)}")
    energy_spacer_3_UA = fc_spacer_3_UA.mfe()[1]

    ref_seq_spacer_4GC = 'GCGC'
    ref_seq_spacer_4UA = 'UAUA'
    fc_spacer_4_GC = RNA.fold_compound(f"{ref_seq_spacer_4GC}&{RNA_reverse_complement(ref_seq_spacer_4GC)}")
    energy_spacer_4_GC = fc_spacer_4_GC.mfe()[1]
    fc_spacer_4_UA = RNA.fold_compound(f"{ref_seq_spacer_4UA}&{RNA_reverse_complement(ref_seq_spacer_4UA)}")
    energy_spacer_4_UA = fc_spacer_4_UA.mfe()[1]

    ref_seq_spacer_5GC = 'GCGCG'
    ref_seq_spacer_5UA = 'UAUAU'
    fc_spacer_5_GC = RNA.fold_compound(f"{ref_seq_spacer_5GC}&{RNA_reverse_complement(ref_seq_spacer_5GC)}")
    energy_spacer_5_GC = fc_spacer_5_GC.mfe()[1]
    fc_spacer_5_UA = RNA.fold_compound(f"{ref_seq_spacer_5UA}&{RNA_reverse_complement(ref_seq_spacer_5UA)}")
    energy_spacer_5_UA = fc_spacer_5_UA.mfe()[1]

    ref_seq_spacer_6GC = 'GCGCGC'
    ref_seq_spacer_6UA = 'UAUAUA'
    fc_spacer_6_GC = RNA.fold_compound(f"{ref_seq_spacer_6GC}&{RNA_reverse_complement(ref_seq_spacer_6GC)}")
    energy_spacer_6_GC = fc_spacer_6_GC.mfe()[1]
    fc_spacer_6_UA = RNA.fold_compound(f"{ref_seq_spacer_6UA}&{RNA_reverse_complement(ref_seq_spacer_6UA)}")
    energy_spacer_6_UA = fc_spacer_6_UA.mfe()[1]

    ref_seq_spacer_7GC = 'GCGCGCG'
    ref_seq_spacer_7UA = 'UAUAUAU'
    fc_spacer_7_GC = RNA.fold_compound(f"{ref_seq_spacer_7GC}&{RNA_reverse_complement(ref_seq_spacer_7GC)}")
    energy_spacer_7_GC = fc_spacer_7_GC.mfe()[1]
    fc_spacer_7_UA = RNA.fold_compound(f"{ref_seq_spacer_7UA}&{RNA_reverse_complement(ref_seq_spacer_7UA)}")
    energy_spacer_7_UA = fc_spacer_7_UA.mfe()[1]

    ref_seq_spacer_8GC = 'GCGCGCGC' 
    ref_seq_spacer_8UA = 'UAUAUAUA'
    fc_spacer_8_GC = RNA.fold_compound(f"{ref_seq_spacer_8GC}&{RNA_reverse_complement(ref_seq_spacer_8GC)}")
    energy_spacer_8_GC = fc_spacer_8_GC.mfe()[1]
    fc_spacer_8_UA = RNA.fold_compound(f"{ref_seq_spacer_8UA}&{RNA_reverse_complement(ref_seq_spacer_8UA)}")
    energy_spacer_8_UA = fc_spacer_8_UA.mfe()[1]
    
    fc_scaffold = RNA.fold_compound(f"{scaffold_seq}")
    energy_scaffold = fc_scaffold.mfe()[1]        

    # Replace all of these with your real calculation functions
    if mode == "Cas9":
        RNA.params_load_RNA_Turner2004()
        full_guide_GC = DNA_to_RNA(ref_seq_full_GC) + scaffold_seq 
        full_guide_TA = DNA_to_RNA(ref_seq_full_TA) + scaffold_seq 

        # Calculate guide_min and guide_max
        fc_guide_GC = RNA.fold_compound(f"{full_guide_GC}")
        energy_guide_GC = fc_guide_GC.mfe()[1]
        guide_min = energy_guide_GC - energy_scaffold
        guide_max = energy_scaffold - energy_scaffold

        # Calculate target_min and target_max
        RNA.params_load_DNA_Mathews2004()
        fc_duplex_GC = RNA.fold_compound(f"{full_guide_GC}&{DNA_reverse_complement(ref_seq_full_GC)}")
        energy_duplex_GC = fc_duplex_GC.mfe()[1]
        fc_duplex_TA = RNA.fold_compound(f"{full_guide_TA}&{DNA_reverse_complement(ref_seq_full_TA)}")
        energy_duplex_TA = fc_duplex_TA.mfe()[1]
        target_min = energy_duplex_GC
        target_max = energy_duplex_TA

        # Calculate ss_target_bh_min and ss_target_bh_max
        fc_ss_target_GC = RNA.fold_compound(f"{ref_seq_full_GC}")
        energy_ss_target_GC = fc_ss_target_GC.mfe()[1]
        ss_target_bh_min = energy_ss_target_GC
        ss_target_bh_max = 0

        # Calculate guide_conse_min and guide_conse_max
        RNA.params_load_RNA_Turner2004()
        fc_ds_target_GC_RNA = RNA.fold_compound(f"{ref_seq_full_GC}&{RNA_reverse_complement(ref_seq_full_GC)}")
        energy_ds_target_GC_RNA = fc_ds_target_GC_RNA.mfe()[1]
        fc_ds_target_single_A_RNA = RNA.fold_compound(f"{ref_seq_single_A}&{RNA_reverse_complement(ref_seq_single_A)}")
        energy_ds_target_single_A_RNA = fc_ds_target_single_A_RNA.mfe()[1]
        guide_conse_min = energy_ds_target_GC_RNA
        guide_conse_max = energy_ds_target_single_A_RNA

        # Calculate target_conse_min and target_conse_max
        RNA.params_load_DNA_Mathews2004()
        fc_ds_target_GC_DNA = RNA.fold_compound(f"{ref_seq_full_GC}&{DNA_reverse_complement(ref_seq_full_GC)}")
        energy_ds_target_GC_DNA = fc_ds_target_GC_DNA.mfe()[1]
        fc_ds_target_single_A_DNA = RNA.fold_compound(f"{ref_seq_single_A}&{DNA_reverse_complement(ref_seq_single_A)}")
        energy_ds_target_single_A_DNA = fc_ds_target_single_A_DNA.mfe()[1]
        target_conse_min = energy_ds_target_GC_DNA
        target_conse_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_conse_min and ss_target_bh_conse_max
        fc_ref_seq_conse_GC = RNA.fold_compound(f"{ref_seq_conse_GC}&{DNA_reverse_complement(ref_seq_conse_GC)}")
        energy_ref_seq_conse_GC = fc_ref_seq_conse_GC.mfe()[1]
        ss_target_bh_conse_min = energy_ref_seq_conse_GC
        ss_target_bh_conse_max = 0

        # Calculate guide_conse_unpaired_min and guide_conse_unpaired_max
        guide_conse_unpaired_min = energy_ds_target_GC_RNA
        guide_conse_unpaired_max = energy_ds_target_single_A_RNA

        # Calculate target_conse_unpaired_min and target_conse_unpaired_max
        target_conse_unpaired_min = energy_ds_target_GC_DNA
        target_conse_unpaired_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_conse_unpaired_min and ss_target_bh_conse_unpaired_max
        fc_ref_seq_full_G = RNA.fold_compound(f"{ref_seq_full_G}&{DNA_reverse_complement(ref_seq_full_G)}")
        energy_ref_seq_full_G = fc_ref_seq_full_G.mfe()[1]
        ss_target_bh_conse_unpaired_min = energy_ref_seq_full_G
        ss_target_bh_conse_unpaired_max = energy_ds_target_single_A_DNA

        # Calculate guide_overhang_min and guide_overhang_max
        guide_overhang_min = energy_ds_target_GC_RNA
        guide_overhang_max = energy_ds_target_single_A_RNA

        # Calculate target_overhang_min and target_overhang_max
        target_overhang_min = energy_ds_target_GC_DNA
        target_overhang_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_overhang_min and ss_target_bh_overhang_max
        ss_target_bh_overhang_min = energy_ref_seq_full_G
        ss_target_bh_overhang_max = energy_ds_target_single_A_DNA
     
        # Calculate guide_paired_min and guide_paired_max
        guide_paired_min = energy_ds_target_GC_RNA
        guide_paired_max = energy_ds_target_single_A_RNA

        # Calculate target_paired_min and target_paired_max
        target_paired_min = energy_ds_target_GC_DNA
        target_paired_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_paired_min and ss_target_bh_paired_max
        ss_target_bh_paired_min = energy_ref_seq_conse_GC
        ss_target_bh_paired_max = 0

        # Calculate seed_min and seed_max
        fc_seed_GC = RNA.fold_compound(f"{ref_seq_full_GC_seed}&{DNA_reverse_complement(ref_seq_full_GC_seed)}")
        energy_seed_GC = fc_seed_GC.mfe()[1]
        fc_seed_TA = RNA.fold_compound(f"{ref_seq_full_TA_seed}&{DNA_reverse_complement(ref_seq_full_TA_seed)}")
        energy_seed_TA = fc_seed_TA.mfe()[1]
        seed_min = energy_seed_GC
        seed_max = energy_seed_TA

        # Calculate middle_min and middle_max
        fc_middle_GC = RNA.fold_compound(f"{ref_seq_full_GC_middle}&{DNA_reverse_complement(ref_seq_full_GC_middle)}")
        energy_middle_GC = fc_middle_GC.mfe()[1]
        fc_middle_TA = RNA.fold_compound(f"{ref_seq_full_TA_middle}&{DNA_reverse_complement(ref_seq_full_TA_middle)}")
        energy_middle_TA = fc_middle_TA.mfe()[1]
        middle_min = energy_middle_GC
        middle_max = energy_middle_TA

        # Calculate distal_min and distal_max
        fc_distal_GC = RNA.fold_compound(f"{ref_seq_full_GC_distal}&{DNA_reverse_complement(ref_seq_full_GC_distal)}")
        energy_distal_GC = fc_distal_GC.mfe()[1]
        fc_distal_TA = RNA.fold_compound(f"{ref_seq_full_TA_distal}&{DNA_reverse_complement(ref_seq_full_TA_distal)}")
        energy_distal_TA = fc_distal_TA.mfe()[1]
        distal_min = energy_distal_GC
        distal_max = energy_distal_TA

        # Calculate target_conse_3_min and target_conse_3_max
        target_conse_3_min = energy_spacer_3_GC
        target_conse_3_max = energy_spacer_3_UA

        # Calculate target_conse_4_min and target_conse_4_max
        target_conse_4_min = energy_spacer_4_GC
        target_conse_4_max = energy_spacer_4_UA

        # Calculate target_conse_5_min and target_conse_5_max
        target_conse_5_min = energy_spacer_5_GC
        target_conse_5_max = energy_spacer_5_UA

        # Calculate target_conse_6_min and target_conse_6_max
        target_conse_6_min = energy_spacer_6_GC
        target_conse_6_max = energy_spacer_6_UA

        # Calculate target_conse_7_min and target_conse_7_max
        target_conse_7_min = energy_spacer_7_GC
        target_conse_7_max = energy_spacer_7_UA

        # Calculate target_conse_8_min and target_conse_8_max
        target_conse_8_min = energy_spacer_8_GC
        target_conse_8_max = energy_spacer_8_UA

    elif mode == "Cas12":  # Cas12
        RNA.params_load_RNA_Turner2004()
        full_guide_GC = scaffold_seq + DNA_to_RNA(ref_seq_full_GC)
        full_guide_TA = scaffold_seq + DNA_to_RNA(ref_seq_full_TA)

        # Calculate guide_min and guide_max
        fc_guide_GC = RNA.fold_compound(f"{full_guide_GC}")
        energy_guide_GC = fc_guide_GC.mfe()[1]
        guide_min = energy_guide_GC - energy_scaffold
        guide_max = energy_scaffold - energy_scaffold

        # Calculate target_min and target_max
        RNA.params_load_DNA_Mathews2004()
        fc_duplex_GC = RNA.fold_compound(f"{full_guide_GC}&{DNA_reverse_complement(ref_seq_full_GC)}")
        energy_duplex_GC = fc_duplex_GC.mfe()[1]
        fc_duplex_TA = RNA.fold_compound(f"{full_guide_TA}&{DNA_reverse_complement(ref_seq_full_TA)}")
        energy_duplex_TA = fc_duplex_TA.mfe()[1]
        target_min = energy_duplex_GC
        target_max = energy_duplex_TA

        # Calculate ss_target_bh_min and ss_target_bh_max
        fc_ss_target_GC = RNA.fold_compound(f"{ref_seq_full_GC}")
        energy_ss_target_GC = fc_ss_target_GC.mfe()[1]
        ss_target_bh_min = energy_ss_target_GC
        ss_target_bh_max = 0

        # Calculate guide_conse_min and guide_conse_max
        RNA.params_load_RNA_Turner2004()
        fc_ds_target_GC_RNA = RNA.fold_compound(f"{ref_seq_full_GC}&{RNA_reverse_complement(ref_seq_full_GC)}")
        energy_ds_target_GC_RNA = fc_ds_target_GC_RNA.mfe()[1]
        fc_ds_target_single_A_RNA = RNA.fold_compound(f"{ref_seq_single_A}&{RNA_reverse_complement(ref_seq_single_A)}")
        energy_ds_target_single_A_RNA = fc_ds_target_single_A_RNA.mfe()[1]
        guide_conse_min = energy_ds_target_GC_RNA
        guide_conse_max = energy_ds_target_single_A_RNA

        # Calculate target_conse_min and target_conse_max
        RNA.params_load_DNA_Mathews2004()
        fc_ds_target_GC_DNA = RNA.fold_compound(f"{ref_seq_full_GC}&{DNA_reverse_complement(ref_seq_full_GC)}")
        energy_ds_target_GC_DNA = fc_ds_target_GC_DNA.mfe()[1]
        fc_ds_target_single_A_DNA = RNA.fold_compound(f"{ref_seq_single_A}&{DNA_reverse_complement(ref_seq_single_A)}")
        energy_ds_target_single_A_DNA = fc_ds_target_single_A_DNA.mfe()[1]
        target_conse_min = energy_ds_target_GC_DNA
        target_conse_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_conse_min and ss_target_bh_conse_max
        fc_ref_seq_conse_GC = RNA.fold_compound(f"{ref_seq_conse_GC}&{DNA_reverse_complement(ref_seq_conse_GC)}")
        energy_ref_seq_conse_GC = fc_ref_seq_conse_GC.mfe()[1]
        ss_target_bh_conse_min = energy_ref_seq_conse_GC
        ss_target_bh_conse_max = 0

        # Calculate guide_conse_unpaired_min and guide_conse_unpaired_max
        guide_conse_unpaired_min = energy_ds_target_GC_RNA
        guide_conse_unpaired_max = energy_ds_target_single_A_RNA

        # Calculate target_conse_unpaired_min and target_conse_unpaired_max
        target_conse_unpaired_min = energy_ds_target_GC_DNA
        target_conse_unpaired_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_conse_unpaired_min and ss_target_bh_conse_unpaired_max
        fc_ref_seq_full_G = RNA.fold_compound(f"{ref_seq_full_G}&{DNA_reverse_complement(ref_seq_full_G)}")
        energy_ref_seq_full_G = fc_ref_seq_full_G.mfe()[1]
        ss_target_bh_conse_unpaired_min = energy_ref_seq_full_G
        ss_target_bh_conse_unpaired_max = energy_ds_target_single_A_DNA

        # Calculate guide_overhang_min and guide_overhang_max
        guide_overhang_min = energy_ds_target_GC_RNA
        guide_overhang_max = energy_ds_target_single_A_RNA

        # Calculate target_overhang_min and target_overhang_max
        target_overhang_min = energy_ds_target_GC_DNA
        target_overhang_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_overhang_min and ss_target_bh_overhang_max
        ss_target_bh_overhang_min = energy_ref_seq_full_G
        ss_target_bh_overhang_max = energy_ds_target_single_A_DNA

        # Calculate guide_paired_min and guide_paired_max
        guide_paired_min = energy_ds_target_GC_RNA
        guide_paired_max = energy_ds_target_single_A_RNA

        # Calculate target_paired_min and target_paired_max
        target_paired_min = energy_ds_target_GC_DNA
        target_paired_max = energy_ds_target_single_A_DNA

        # Calculate ss_target_bh_paired_min and ss_target_bh_paired_max
        ss_target_bh_paired_min = energy_ref_seq_conse_GC
        ss_target_bh_paired_max = 0

        # Calculate seed_min and seed_max
        fc_seed_GC = RNA.fold_compound(f"{ref_seq_full_GC_seed}&{DNA_reverse_complement(ref_seq_full_GC_seed)}")
        energy_seed_GC = fc_seed_GC.mfe()[1]
        fc_seed_TA = RNA.fold_compound(f"{ref_seq_full_TA_seed}&{DNA_reverse_complement(ref_seq_full_TA_seed)}")
        energy_seed_TA = fc_seed_TA.mfe()[1]
        seed_min = energy_seed_GC
        seed_max = energy_seed_TA

        # Calculate middle_min and middle_max
        fc_middle_GC = RNA.fold_compound(f"{ref_seq_full_GC_middle}&{DNA_reverse_complement(ref_seq_full_GC_middle)}")
        energy_middle_GC = fc_middle_GC.mfe()[1]
        fc_middle_TA = RNA.fold_compound(f"{ref_seq_full_TA_middle}&{DNA_reverse_complement(ref_seq_full_TA_middle)}")
        energy_middle_TA = fc_middle_TA.mfe()[1]
        middle_min = energy_middle_GC
        middle_max = energy_middle_TA

        # Calculate distal_min and distal_max
        fc_distal_GC = RNA.fold_compound(f"{ref_seq_full_GC_distal}&{DNA_reverse_complement(ref_seq_full_GC_distal)}")
        energy_distal_GC = fc_distal_GC.mfe()[1]
        fc_distal_TA = RNA.fold_compound(f"{ref_seq_full_TA_distal}&{DNA_reverse_complement(ref_seq_full_TA_distal)}")
        energy_distal_TA = fc_distal_TA.mfe()[1]
        distal_min = energy_distal_GC
        distal_max = energy_distal_TA

        # Calculate target_conse_3_min and target_conse_3_max
        target_conse_3_min = energy_spacer_3_GC
        target_conse_3_max = energy_spacer_3_UA

        # Calculate target_conse_4_min and target_conse_4_max
        target_conse_4_min = energy_spacer_4_GC
        target_conse_4_max = energy_spacer_4_UA

        # Calculate target_conse_5_min and target_conse_5_max
        target_conse_5_min = energy_spacer_5_GC
        target_conse_5_max = energy_spacer_5_UA

        # Calculate target_conse_6_min and target_conse_6_max
        target_conse_6_min = energy_spacer_6_GC
        target_conse_6_max = energy_spacer_6_UA

        # Calculate target_conse_7_min and target_conse_7_max
        target_conse_7_min = energy_spacer_7_GC
        target_conse_7_max = energy_spacer_7_UA

        # Calculate target_conse_8_min and target_conse_8_max
        target_conse_8_min = energy_spacer_8_GC
        target_conse_8_max = energy_spacer_8_UA
    
    elif mode == "Cas13":  # Cas13
        RNA.params_load_RNA_Turner2004()
        full_guide_GC = scaffold_seq + DNA_to_RNA(ref_seq_full_GC)
        full_guide_UA = scaffold_seq + DNA_to_RNA(ref_seq_full_UA)

        # Calculate guide_min and guide_max
        fc_guide_GC = RNA.fold_compound(f"{full_guide_GC}")
        energy_guide_GC = fc_guide_GC.mfe()[1]
        guide_min = energy_guide_GC - energy_scaffold
        guide_max = energy_scaffold - energy_scaffold

        # Calculate target_min and target_max
        fc_duplex_GC = RNA.fold_compound(f"{full_guide_GC}&{RNA_reverse_complement(ref_seq_full_GC)}")
        energy_duplex_GC = fc_duplex_GC.mfe()[1]
        fc_duplex_UA = RNA.fold_compound(f"{full_guide_UA}&{RNA_reverse_complement(ref_seq_full_UA)}")
        energy_duplex_UA = fc_duplex_UA.mfe()[1]
        target_min = energy_duplex_GC
        target_max = energy_duplex_UA

        # Calculate ss_target_bh_min and ss_target_bh_max
        fc_ss_target_GC = RNA.fold_compound(f"{ref_seq_full_GC}")
        energy_ss_target_GC = fc_ss_target_GC.mfe()[1]
        ss_target_bh_min = energy_ss_target_GC
        ss_target_bh_max = 0

        # Calculate guide_conse_min and guide_conse_max
        fc_ds_target_GC_RNA = RNA.fold_compound(f"{ref_seq_full_GC}&{RNA_reverse_complement(ref_seq_full_GC)}")
        energy_ds_target_GC_RNA = fc_ds_target_GC_RNA.mfe()[1]
        fc_ds_target_single_A_RNA = RNA.fold_compound(f"{ref_seq_single_A}&{RNA_reverse_complement(ref_seq_single_A)}")
        energy_ds_target_single_A_RNA = fc_ds_target_single_A_RNA.mfe()[1]
        guide_conse_min = energy_ds_target_GC_RNA
        guide_conse_max = energy_ds_target_single_A_RNA

        # Calculate target_conse_min and target_conse_max
        target_conse_min = energy_ds_target_GC_RNA
        target_conse_max = energy_ds_target_single_A_RNA

        # Calculate ss_target_bh_conse_min and ss_target_bh_conse_max
        fc_ref_seq_conse_GC = RNA.fold_compound(f"{ref_seq_conse_GC}&{RNA_reverse_complement(ref_seq_conse_GC)}")
        energy_ref_seq_conse_GC = fc_ref_seq_conse_GC.mfe()[1]
        ss_target_bh_conse_min = energy_ref_seq_conse_GC
        ss_target_bh_conse_max = 0

        # Calculate guide_conse_unpaired_min and guide_conse_unpaired_max
        guide_conse_unpaired_min = energy_ds_target_GC_RNA
        guide_conse_unpaired_max = energy_ds_target_single_A_RNA

        # Calculate target_conse_unpaired_min and target_conse_unpaired_max
        target_conse_unpaired_min = energy_ds_target_GC_RNA
        target_conse_unpaired_max = energy_ds_target_single_A_RNA

        # Calculate ss_target_bh_conse_unpaired_min and ss_target_bh_conse_unpaired_max
        fc_ref_seq_full_G = RNA.fold_compound(f"{ref_seq_full_G}&{RNA_reverse_complement(ref_seq_full_G)}")
        energy_ref_seq_full_G = fc_ref_seq_full_G.mfe()[1]
        ss_target_bh_conse_unpaired_min = energy_ref_seq_full_G
        ss_target_bh_conse_unpaired_max = energy_ds_target_single_A_RNA

        # Calculate guide_overhang_min and guide_overhang_max
        guide_overhang_min = energy_ds_target_GC_RNA
        guide_overhang_max = energy_ds_target_single_A_RNA

        # Calculate target_overhang_min and target_overhang_max
        target_overhang_min = energy_ds_target_GC_RNA
        target_overhang_max = energy_ds_target_single_A_RNA

        # Calculate ss_target_bh_overhang_min and ss_target_bh_overhang_max
        ss_target_bh_overhang_min = energy_ref_seq_full_G
        ss_target_bh_overhang_max = energy_ds_target_single_A_RNA
        
        # Calculate guide_paired_min and guide_paired_max
        guide_paired_min = energy_ds_target_GC_RNA
        guide_paired_max = energy_ds_target_single_A_RNA

        # Calculate target_paired_min and target_paired_max
        target_paired_min = energy_ds_target_GC_RNA
        target_paired_max = energy_ds_target_single_A_RNA

        # Calculate ss_target_bh_paired_min and ss_target_bh_paired_max
        ss_target_bh_paired_min = energy_ref_seq_conse_GC
        ss_target_bh_paired_max = 0

        # Calculate seed_min and seed_max
        fc_seed_GC = RNA.fold_compound(f"{ref_seq_full_GC_seed}&{RNA_reverse_complement(ref_seq_full_GC_seed)}")
        energy_seed_GC = fc_seed_GC.mfe()[1]
        fc_seed_UA = RNA.fold_compound(f"{ref_seq_full_UA_seed}&{RNA_reverse_complement(ref_seq_full_UA_seed)}")
        energy_seed_UA = fc_seed_UA.mfe()[1]
        seed_min = energy_seed_GC
        seed_max = energy_seed_UA

        # Calculate middle_min and middle_max
        fc_middle_GC = RNA.fold_compound(f"{ref_seq_full_GC_middle}&{RNA_reverse_complement(ref_seq_full_GC_middle)}")
        energy_middle_GC = fc_middle_GC.mfe()[1]
        fc_middle_UA = RNA.fold_compound(f"{ref_seq_full_UA_middle}&{RNA_reverse_complement(ref_seq_full_UA_middle)}")
        energy_middle_UA = fc_middle_UA.mfe()[1]
        middle_min = energy_middle_GC
        middle_max = energy_middle_UA

        # Calculate distal_min and distal_max
        fc_distal_GC = RNA.fold_compound(f"{ref_seq_full_GC_distal}&{RNA_reverse_complement(ref_seq_full_GC_distal)}")
        energy_distal_GC = fc_distal_GC.mfe()[1]
        fc_distal_UA = RNA.fold_compound(f"{ref_seq_full_UA_distal}&{RNA_reverse_complement(ref_seq_full_UA_distal)}")
        energy_distal_UA = fc_distal_UA.mfe()[1]
        distal_min = energy_distal_GC
        distal_max = energy_distal_UA

        # Calculate target_conse_3_min and target_conse_3_max
        target_conse_3_min = energy_spacer_3_GC
        target_conse_3_max = energy_spacer_3_UA

        # Calculate target_conse_4_min and target_conse_4_max
        target_conse_4_min = energy_spacer_4_GC
        target_conse_4_max = energy_spacer_4_UA

        # Calculate target_conse_5_min and target_conse_5_max
        target_conse_5_min = energy_spacer_5_GC
        target_conse_5_max = energy_spacer_5_UA

        # Calculate target_conse_6_min and target_conse_6_max
        target_conse_6_min = energy_spacer_6_GC
        target_conse_6_max = energy_spacer_6_UA

        # Calculate target_conse_7_min and target_conse_7_max
        target_conse_7_min = energy_spacer_7_GC
        target_conse_7_max = energy_spacer_7_UA

        # Calculate target_conse_8_min and target_conse_8_max
        target_conse_8_min = energy_spacer_8_GC
        target_conse_8_max = energy_spacer_8_UA

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