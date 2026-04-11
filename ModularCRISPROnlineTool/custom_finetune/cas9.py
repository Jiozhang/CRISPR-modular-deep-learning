import numpy as np
import torch
from torch_geometric.data import Batch

from .parameter_calculator import calculate_custom_feature_params, clean_seq
from . import features_cas9 as feat

def DNA_reverse_complement(DNA):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(DNA))

def validate_pair(guide: str, target: str):
    allowed = set("ACGTU")
    if len(guide) == 0 or len(target) == 0:
        raise ValueError("Guide and target cannot be empty.")
    if len(guide) > 30 or len(target) > 30:
        raise ValueError("Guide and target must each be at most 30 nt.")
    bad = [c for c in guide + target if c not in allowed]
    if bad:
        raise ValueError(f"Invalid bases found: {sorted(set(bad))}")


def build_training_components_from_dataframe(df, scaffold_seq: str, spacer_len: int):
    params = calculate_custom_feature_params(scaffold_seq, spacer_len, mode="Cas9")

    guides = [clean_seq(x) for x in df.iloc[:, 0].tolist()]
    targets = [clean_seq(x) for x in df.iloc[:, 1].tolist()]
    y = np.asarray(df.iloc[:, 2].astype(float).tolist(), dtype=np.float32)

    X1, X2, X3 = [], [], []
    guide_graphs, target_graphs, duplex_graphs = [], [], []

    for g, t in zip(guides, targets):
        
        t = DNA_reverse_complement(t)
        validate_pair(g, t)

        X1.append(feat.build_sequence_features(g, t))
        X2.append(feat.build_structure_features(g, t, params))
        X3.append(feat.build_energy_features(g, t, params))

        guide_graphs.append(feat.build_guide_graph(g, params))
        target_graphs.append(feat.build_target_bh_graph(t))
        duplex_graphs.append(feat.build_duplex_graph(g, t, params))

    return guide_graphs, target_graphs, duplex_graphs, np.asarray(X1), np.asarray(X2), np.asarray(X3), y, params


def predict_single(model, guide: str, target: str, scaffold_seq: str, spacer_len: int, device="cpu"):
    params = calculate_custom_feature_params(scaffold_seq, spacer_len, mode="Cas9")

    guide = clean_seq(guide)
    target = clean_seq(target)
    target = DNA_reverse_complement(target)
    validate_pair(guide, target)

    x1 = torch.tensor(feat.build_sequence_features(guide, target), dtype=torch.float32).unsqueeze(0).to(device)
    x2 = torch.tensor(feat.build_structure_features(guide, target, params), dtype=torch.float32).unsqueeze(0).to(device)
    x3 = torch.tensor(feat.build_energy_features(guide, target, params), dtype=torch.float32).unsqueeze(0).to(device)

    guide_data = feat.build_guide_graph(guide, params)
    target_data = feat.build_target_bh_graph(target)
    duplex_data = feat.build_duplex_graph(guide, target, params)

    batch_guide = Batch.from_data_list([guide_data]).to(device)
    batch_target = Batch.from_data_list([target_data]).to(device)
    batch_duplex = Batch.from_data_list([duplex_data]).to(device)

    with torch.no_grad():
        pred = model(batch_guide, batch_target, batch_duplex, x1, x2, x3)

    return float(pred.squeeze().cpu().item())