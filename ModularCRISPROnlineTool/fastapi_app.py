import os
import json
import uuid
import copy
import traceback
import importlib
import __main__
from typing import Dict, Any
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
import numpy as np
import optuna


# shared ONLY for fine-tuned models
from fine_tuned.shared.model_defs import (
    CNNBranch1,
    CNNBranch2,
    MLPBranch120,
    GNNBranch,
    CNN_GNN_MLP_Fusion,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Modular CRISPR Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
USER_MODEL_DIR = "user_models"
GENERALIST_MODEL_PATH = "generalist_model.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(USER_MODEL_DIR, exist_ok=True)

def register_fine_tuned_shared_classes():
    __main__.CNNBranch1 = CNNBranch1
    __main__.CNNBranch2 = CNNBranch2
    __main__.MLPBranch120 = MLPBranch120
    __main__.GNNBranch = GNNBranch
    __main__.CNN_GNN_MLP_Fusion = CNN_GNN_MLP_Fusion

def build_custom_model_from_trial(trial):
    filters1       = trial.suggest_int("filters1", 32, 128, step=32)
    kernel_size1   = trial.suggest_categorical("kernel_size1", [3, 5, 7])
    dense_units1   = trial.suggest_int("dense_units1", 64, 256, step=64)

    filters2       = trial.suggest_int("filters2", 32, 128, step=32)
    kernel_size2   = trial.suggest_categorical("kernel_size2", [3, 5, 7])
    dense_units2   = trial.suggest_int("dense_units2", 64, 256, step=64)

    gnn_hidden_dim = trial.suggest_int("gnn_hidden_dim", 32, 128, step=32)
    mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 32, 128, step=32)
    final_fc_dim   = trial.suggest_int("final_fc_dim", 64, 256, step=64)
    dropout_rate   = trial.suggest_float("dropout_rate", 0.0, 0.5)

    model = CNN_GNN_MLP_Fusion(
        filters1=filters1,
        kernel_size1=kernel_size1,
        dense_units1=dense_units1,
        filters2=filters2,
        kernel_size2=kernel_size2,
        dense_units2=dense_units2,
        gnn_hidden_dim=gnn_hidden_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        final_fc_dim=final_fc_dim,
        dropout_rate=dropout_rate,
    )
    return model

INDIVIDUAL_REACTION_CONFIGS: Dict[str, Dict[str, str]] = {
    "SpCas9_Kim_2019_SA": {
        "key": "SpCas9_Kim_2019_SA",
        "package": "reactions.SpCas9_Kim_2019_SA",
        "model_path": "reactions/SpCas9_Kim_2019_SA/best_FULL.pt",
    },
    "SpCas9_Wang_2019_NC": {
        "key": "SpCas9_Wang_2019_NC",
        "package": "reactions.SpCas9_Wang_2019_NC",
        "model_path": "reactions/SpCas9_Wang_2019_NC/best_FULL.pt",
    },
    "SpCas9-mismatch_Doench_2016_NB": {
        "key": "SpCas9-mismatch_Doench_2016_NB",
        "package": "reactions.SpCas9-mismatch_Doench_2016_NB",
        "model_path": "reactions/SpCas9-mismatch_Doench_2016_NB/best_FULL.pt",
    },
    "SpCas9-mismatch_Kim_2020_NB": {
        "key": "SpCas9-mismatch_Kim_2020_NB",
        "package": "reactions.SpCas9-mismatch_Kim_2020_NB",
        "model_path": "reactions/SpCas9-mismatch_Kim_2020_NB/best_FULL.pt",
    },
    "SpCas9-NG_Kim_2020_NB": {
        "key": "SpCas9-NG_Kim_2020_NB",
        "package": "reactions.SpCas9-NG_Kim_2020_NB",
        "model_path": "reactions/SpCas9-NG_Kim_2020_NB/best_FULL.pt",
    },
    "SpCas9-HF1_Kim_2020_NB": {
        "key": "SpCas9-HF1_Kim_2020_NB",
        "package": "reactions.SpCas9-HF1_Kim_2020_NB",
        "model_path": "reactions/SpCas9-HF1_Kim_2020_NB/best_FULL.pt",
    },
    "eSpCas9_Kim_2020_NB": {
        "key": "eSpCas9_Kim_2020_NB",
        "package": "reactions.eSpCas9_Kim_2020_NB",
        "model_path": "reactions/eSpCas9_Kim_2020_NB/best_FULL.pt",
    }, 
    "CHANGE-seq-mismatch_Lazzarotto_2020_NB": {
        "key": "CHANGE-seq-mismatch_Lazzarotto_2020_NB",
        "package": "reactions.CHANGE-seq-mismatch_Lazzarotto_2020_NB",
        "model_path": "reactions/CHANGE-seq-mismatch_Lazzarotto_2020_NB/best_FULL.pt",
    },
    "AsCas12a_Kim_2018_NB": {
        "key": "AsCas12a_Kim_2018_NB",
        "package": "reactions.AsCas12a_Kim_2018_NB",
        "model_path": "reactions/AsCas12a_Kim_2018_NB/best_FULL.pt",
    }, 
    "LbCas12aTrans-mismatch_Huang_2024_iMeta": {
        "key": "LbCas12aTrans-mismatch_Huang_2024_iMeta",
        "package": "reactions.LbCas12aTrans-mismatch_Huang_2024_iMeta",
        "model_path": "reactions/LbCas12aTrans-mismatch_Huang_2024_iMeta/best_FULL.pt",
    },
    "dFnCas12a-mismatch_Specht_2020_PNAS": {
        "key": "dFnCas12a-mismatch_Specht_2020_PNAS",
        "package": "reactions.dFnCas12a-mismatch_Specht_2020_PNAS",
        "model_path": "reactions/dFnCas12a-mismatch_Specht_2020_PNAS/best_FULL.pt",
    },
     "SpCas9-NG-YE1-BE4_Kim_2024_NB": {
        "key": "SpCas9-NG-YE1-BE4_Kim_2024_NB",
        "package": "reactions.SpCas9-NG-YE1-BE4_Kim_2024_NB",
        "model_path": "reactions/SpCas9-NG-YE1-BE4_Kim_2024_NB/best_FULL.pt",
    },
    "RfxCas13d-mismatch_Wessels_2024_NB": {
        "key": "RfxCas13d-mismatch_Wessels_2024_NB",
        "package": "reactions.RfxCas13d-mismatch_Wessels_2024_NB",
        "model_path": "reactions/RfxCas13d-mismatch_Wessels_2024_NB/best_FULL.pt",
    },
    "PE2-mismatch_Kim_2021_NB": {
        "key": "PE2-mismatch_Kim_2021_NB",
        "package": "reactions.PE2-mismatch_Kim_2021_NB",
        "model_path": "reactions/PE2-mismatch_Kim_2021_NB/best_FULL.pt",
    },
}

FINE_TUNED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Cas9 variants": {
        "category_key": "Cas9_variants",
        "reactions": {
            "eSpCas9": {
                "key": "eSpCas9",
                "package": "fine_tuned.Cas9_variants.eSpCas9",
                "model_path": "fine_tuned/Cas9_variants/eSpCas9/fine_tuned_200_samples.pt",
            },
            "evoCas9": {
                "key": "evoCas9",
                "package": "fine_tuned.Cas9_variants.evoCas9",
                "model_path": "fine_tuned/Cas9_variants/evoCas9/fine_tuned_200_samples.pt",
            },
            'HypaCas9': {
                "key": "HypaCas9",
                "package": "fine_tuned.Cas9_variants.HypaCas9",
                "model_path": "fine_tuned/Cas9_variants/HypaCas9/fine_tuned_200_samples.pt",
            },
            'Sniper-Cas9': {
                "key": "Sniper-Cas9",
                "package": "fine_tuned.Cas9_variants.Sniper_Cas9",
                "model_path": "fine_tuned/Cas9_variants/Sniper_Cas9/fine_tuned_200_samples.pt",
            },
            'SpCas9-HF1': {
                "key": "SpCas9-HF1",
                "package": "fine_tuned.Cas9_variants.SpCas9_HF1",
                "model_path": "fine_tuned/Cas9_variants/SpCas9_HF1/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG': {
                "key": "SpCas9-NG",
                "package": "fine_tuned.Cas9_variants.SpCas9_NG",
                "model_path": "fine_tuned/Cas9_variants/SpCas9_NG/fine_tuned_200_samples.pt",
            },
            'VRQR': {
                "key": "VRQR",
                "package": "fine_tuned.Cas9_variants.VRQR",
                "model_path": "fine_tuned/Cas9_variants/VRQR/fine_tuned_200_samples.pt",
            },
            'xCas9':{
                "key": "xCas9",
                "package": "fine_tuned.Cas9_variants.xCas9",
                "model_path": "fine_tuned/Cas9_variants/xCas9/fine_tuned_200_samples.pt",
            },
        },
    },
    "Small Cas9 variants": {
        "category_key": "Small_Cas9_variants",
        "reactions": {
            "efSaCas9": {
                "key": "efSaCas9",
                "package": "fine_tuned.Small_Cas9_variants.efSaCas9",
                "model_path": "fine_tuned/Small_Cas9_variants/efSaCas9/fine_tuned_200_samples.pt",
            },
            "eSaCas9": {
                "key": "eSaCas9",
                "package": "fine_tuned.Small_Cas9_variants.eSaCas9",
                "model_path": "fine_tuned/Small_Cas9_variants/eSaCas9/fine_tuned_200_samples.pt",
            },
            'SaCas9': {
                "key": "SaCas9",
                "package": "fine_tuned.Small_Cas9_variants.SaCas9",
                "model_path": "fine_tuned/Small_Cas9_variants/SaCas9/fine_tuned_200_samples.pt",
            },
            'SaCas9-HF': {
                "key": "SaCas9-HF",
                "package": "fine_tuned.Small_Cas9_variants.SaCas9_HF",
                "model_path": "fine_tuned/Small_Cas9_variants/SaCas9_HF/fine_tuned_200_samples.pt",
            },
            'SaCas9-KKH': {
                "key": "SaCas9-KKH",
                "package": "fine_tuned.Small_Cas9_variants.SaCas9_KKH",
                "model_path": "fine_tuned/Small_Cas9_variants/SaCas9_KKH/fine_tuned_200_samples.pt",
            },
            'SaCas9-KKH-HF': {
                "key": "SaCas9-KKH-HF",
                "package": "fine_tuned.Small_Cas9_variants.SaCas9_KKH_HF",
                "model_path": "fine_tuned/Small_Cas9_variants/SaCas9_KKH_HF/fine_tuned_200_samples.pt",
            },
            'SauriCas9': {
                "key": "SauriCas9",
                "package": "fine_tuned.Small_Cas9_variants.SauriCas9",
                "model_path": "fine_tuned/Small_Cas9_variants/SauriCas9/fine_tuned_200_samples.pt",
            },
            'SauriCas9-KKH': {
                "key": "SauriCas9-KKH",
                "package": "fine_tuned.Small_Cas9_variants.SauriCas9_KKH",
                "model_path": "fine_tuned/Small_Cas9_variants/SauriCas9_KKH/fine_tuned_200_samples.pt",
            },
            'SlugCas9': {
                "key": "SlugCas9",
                "package": "fine_tuned.Small_Cas9_variants.SlugCas9",
                "model_path": "fine_tuned/Small_Cas9_variants/SlugCas9/fine_tuned_200_samples.pt",
            },
            'sRGN3.1': {
                "key": "sRGN3.1",
                "package": "fine_tuned.Small_Cas9_variants.sRGN31",
                "model_path": "fine_tuned/Small_Cas9_variants/sRGN31/fine_tuned_200_samples.pt",
            },
            'St1Cas9': {
                "key": "St1Cas9",
                "package": "fine_tuned.Small_Cas9_variants.St1Cas9",
                "model_path": "fine_tuned/Small_Cas9_variants/St1Cas9/fine_tuned_200_samples.pt",
            },
        },
    },
    "Cas12a variants": {
        "category_key": "Cas12a_variants",
        "reactions": {
            "AsCas12a-Plus": {
                "key": "AsCas12a-Plus",
                "package": "fine_tuned.Cas12a_variants.AsCas12a_Plus",
                "model_path": "fine_tuned/Cas12a_variants/AsCas12a_Plus/fine_tuned_200_samples.pt",
            },
            'AsCas12a-Ultra': {
                "key": "AsCas12a-Ultra",
                "package": "fine_tuned.Cas12a_variants.AsCas12a_Ultra",
                "model_path": "fine_tuned/Cas12a_variants/AsCas12a_Ultra/fine_tuned_200_samples.pt",
            },
            'AsCas12aRR': {
                "key": "AsCas12aRR",
                "package": "fine_tuned.Cas12a_variants.AsCas12aRR",
                "model_path": "fine_tuned/Cas12a_variants/AsCas12aRR/fine_tuned_200_samples.pt",
            },
            'AsCas12aRVR': {
                "key": "AsCas12aRVR",
                "package": "fine_tuned.Cas12a_variants.AsCas12aRVR",
                "model_path": "fine_tuned/Cas12a_variants/AsCas12aRVR/fine_tuned_200_samples.pt",
            },
            'CeCas12a': {
                "key": "CeCas12a",
                "package": "fine_tuned.Cas12a_variants.CeCas12a",
                "model_path": "fine_tuned/Cas12a_variants/CeCas12a/fine_tuned_200_samples.pt",
            },
            'eaFnCas12a': {
                "key": "eaFnCas12a",
                "package": "fine_tuned.Cas12a_variants.eaFnCas12a",
                "model_path": "fine_tuned/Cas12a_variants/eaFnCas12a/fine_tuned_200_samples.pt",
            },
            'EbCas12a': {
                "key": "EbCas12a",
                "package": "fine_tuned.Cas12a_variants.EbCas12a",
                "model_path": "fine_tuned/Cas12a_variants/EbCas12a/fine_tuned_200_samples.pt",
            },
            'enAsCas12a-HF1': {
                "key": "enAsCas12a-HF1",
                "package": "fine_tuned.Cas12a_variants.enAsCas12a_HF1",
                "model_path": "fine_tuned/Cas12a_variants/enAsCas12a_HF1/fine_tuned_200_samples.pt",
            },
            'enEbCas12a': {
                "key": "enEbCas12a",
                "package": "fine_tuned.Cas12a_variants.enEbCas12a",
                "model_path": "fine_tuned/Cas12a_variants/enEbCas12a/fine_tuned_200_samples.pt",
            },
            'FnCas12a': {
                "key": "FnCas12a",
                "package": "fine_tuned.Cas12a_variants.FnCas12a",
                "model_path": "fine_tuned/Cas12a_variants/FnCas12a/fine_tuned_200_samples.pt",
            },
            'FnCas12aRVR': {
                "key": "FnCas12aRVR",
                "package": "fine_tuned.Cas12a_variants.FnCas12a RVR",
                "model_path": "fine_tuned/Cas12a_variants/FnCas12a_RVR/fine_tuned_200_samples.pt",
            },
            'HyperFi-AsCas12a': {
                "key": "HyperFi-AsCas12a",
                "package": "fine_tuned.Cas12a_variants.HyperFi_AsCas12a",
                "model_path": "fine_tuned/Cas12a_variants/HyperFi_AsCas12a/fine_tuned_200_samples.pt",
            },
            'Lb2Cas12a': {
                "key": "Lb2Cas12a",
                "package": "fine_tuned.Cas12a_variants.Lb2Cas12a",
                "model_path": "fine_tuned/Cas12a_variants/Lb2Cas12a/fine_tuned_200_samples.pt",
            },
            'Lb2Cas12aK518R': {
                "key": "Lb2Cas12aK518R",
                "package": "fine_tuned.Cas12a_variants.Lb2Cas12aK518R",
                "model_path": "fine_tuned/Cas12a_variants/Lb2Cas12aK518R/fine_tuned_200_samples.pt",
            },
            'LbCas12a-Plus': {
                "key": "LbCas12a-Plus",
                "package": "fine_tuned.Cas12a_variants.LbCas12a_Plus",
                "model_path": "fine_tuned/Cas12a_variants/LbCas12a_Plus/fine_tuned_200_samples.pt",
            },
            'LbCas12aK538R': {
                "key": "LbCas12aK538R",
                "package": "fine_tuned.Cas12a_variants.LbCas12a K538R",
                "model_path": "fine_tuned/Cas12a_variants/LbCas12aK538R/fine_tuned_200_samples.pt",
            },
            'LbCas12aRR': {
                "key": "LbCas12aRR",
                "package": "fine_tuned.Cas12a_variants.LbCas12aRR",
                "model_path": "fine_tuned/Cas12a_variants/LbCas12aRR/fine_tuned_200_samples.pt",
            },
            'LbCas12aRVR':{
                "key": "LbCas12aRVR",
                "package": "fine_tuned.Cas12a_variants.LbCas12aRVR",
                "model_path": "fine_tuned/Cas12a_variants/LbCas12aRVR/fine_tuned_200_samples.pt",
            },
            'LbCas12aRVRR':{
                "key": "LbCas12aRVRR",
                "package": "fine_tuned.Cas12a_variants.LbCas12aRVRR",
                "model_path": "fine_tuned/Cas12a_variants/LbCas12aRVRR/fine_tuned_200_samples.pt",
            },
            '"mut2C-W': {
                "key": "mut2C-W",
                "package": "fine_tuned.Cas12a_variants.mut2C_W",
                "model_path": "fine_tuned/Cas12a_variants/mut2C_W/fine_tuned_200_samples.pt",
            },
            'mut2C-WF': {
                "key": "mut2C-WF",
                "package": "fine_tuned.Cas12a_variants.mut2C_WF",
                "model_path": "fine_tuned/Cas12a_variants/mut2C_WF/fine_tuned_200_samples.pt",
            },
        },
    },
    "Base editor variants": {
        "category_key": "Base_editor_variants",
        "reactions": {
            "SpCas9-miniCGBE1-3C": {
                "key": "SpCas9-miniCGBE1-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_miniCGBE1_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_miniCGBE1_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-ABE8.17-m+V106W-3A': {
                "key": "SpCas9-NG-ABE8.17-m+V106W-3A",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_ABE817_m_V106W_3A",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_ABE817_m_V106W_3A/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-ABE8e(V106W)-3A': {
                "key": "SpCas9-NG-ABE8e(V106W)-3A",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_ABE8e_V106W_3A",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_ABE8e_V106W_3A/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-APOBEC-nCas9-Ung-3C': {
                "key": "SpCas9-NG-APOBEC-nCas9-Ung-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_APOBEC_nCas9_Ung_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_APOBEC_nCas9_Ung_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-CGBE1-3C': {
                "key": "SpCas9-NG-CGBE1-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_CGBE1_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_CGBE1_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-miniCGBE1-3C': {
                "key": "SpCas9-NG-miniCGBE1-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_miniCGBE1_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_miniCGBE1_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-SsAPOBEC3B-3C': {
                "key": "SpCas9-NG-SsAPOBEC3B-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_SsAPOBEC3B_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_SsAPOBEC3B_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NG-YE1-BE4max_3C': {
                "key": "SpCas9-NG-YE1-BE4max_3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NG_Y E1_BE4max_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NG_YE1_BE4max_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NRCH-ABE8.17-m+V106W-3A': {
                "key": "SpCas9-NRCH-ABE8.17-m+V106W-3A",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NRCH_ABE817_m_V106W_3A",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NRCH_ABE817_m_V106W_3A/fine_tuned_200_samples.pt",
            },
            'SpCas9-NRCH-APOBEC-nCas9-Ung-3C': {
                "key": "SpCas9-NRCH-APOBEC-nCas9-Ung-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NRCH_APOBEC_nCas9_Ung_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NRCH_APOBEC_nCas9_Ung_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NRCH-SsAPOBEC3B-3C': {
                "key": "SpCas9-NRCH-SsAPOBEC3B-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NRCH_SsAPOBEC3B_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NRCH_SsAPOBEC3B_3C/fine_tuned_200_samples.pt",
            },
            'SpCas9-NRCH-YE1-BE4max-3C': {
                "key": "SpCas9-NRCH-YE1-BE4max-3C",
                "package": "fine_tuned.Base_editor_variants.SpCas9_NRCH_YE1_BE4max_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpCas9_NRCH_YE1_BE4max_3C/fine_tuned_200_samples.pt",
            },
            'SpRY-ABE8.17-m+V106W-3A': {
                "key": "SpRY-ABE8.17-m+V106W-3A",
                "package": "fine_tuned.Base_editor_variants.SpRY_ABE817_m_V106W_3A",
                "model_path": "fine_tuned/Base_editor_variants/SpRY_ABE817_m_V106W_3A/fine_tuned_200_samples.pt",
            },
            'SpRY-ABE8e(V106W)-3A': {
                "key": "SpRY-ABE8e(V106W)-3A",
                "package": "fine_tuned.Base_editor_variants.SpRY_ABE8e_V106W_3A",
                "model_path": "fine_tuned/Base_editor_variants/SpRY_ABE8e_V106W_3A/fine_tuned_200_samples.pt",
            },
            'SpRY-YE1-BE4max-3C': {
                "key": "SpRY-YE1-BE4max-3C",
                "package": "fine_tuned.Base_editor_variants.SpRY_YE1_BE4max_3C",
                "model_path": "fine_tuned/Base_editor_variants/SpRY_YE1_BE4max_3C/fine_tuned_200_samples.pt",
            },
        },
    },
    "Prime editor variants": {
        "category_key": "Prime_editor_variants",
        "reactions": {
            "PE2": {
                "key": "PE2",
                "package": "fine_tuned.Prime_editor_variants.PE2",
                "model_path": "fine_tuned/Prime_editor_variants/PE2/fine_tuned_quarter_samples.pt",
            },
            'PE2max': {
                "key": "PE2max",
                "package": "fine_tuned.Prime_editor_variants.PE2max",
                "model_path": "fine_tuned/Prime_editor_variants/PE2max/fine_tuned_quarter_samples.pt",
            },
            'PE4max': {
                "key": "PE4max",
                "package": "fine_tuned.Prime_editor_variants.PE4max",
                "model_path": "fine_tuned/Prime_editor_variants/PE4max/fine_tuned_quarter_samples.pt",
            },
        },
    },
}

# Cache loaded reaction resources so repeated predictions are faster
LOADED_MODELS: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def register_classes_to_main(module):
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            setattr(__main__, name, obj)

def DNA_reverse_complement(DNA):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(DNA))

def RNA_reverse_complement(RNA):
    complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(RNA))

def DNA_to_RNA(DNA):
    match = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'U'}
    return ''.join(match.get(base, base) for base in (DNA))

def slugify(text: str) -> str:
    out = []
    for ch in str(text).strip():
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "custom_reaction"

def transform_output_value(raw_value: float, meta: dict) -> float:
    transform = meta.get("output_transform", {"type": "identity"})
    t = transform.get("type", "identity")

    if t == "identity":
        value = raw_value
    elif t == "divide":
        value = raw_value / float(transform["value"])
    elif t == "multiply":
        value = raw_value * float(transform["value"])
    elif t == "minmax":
        min_val = float(transform["min"])
        max_val = float(transform["max"])
        if max_val == min_val:
            raise ValueError("Invalid minmax transform: max == min")
        value = (raw_value - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported output_transform type: {t}")

    if transform.get("clip_0_1", False):
        value = max(0.0, min(1.0, value))

    return float(value)

# --------------------------------------------------
# Loading predefined resources
# --------------------------------------------------
def load_individual_resources(reaction_key: str) -> Dict[str, Any]:
    cache_key = f"individual::{reaction_key}"
    if cache_key in LOADED_MODELS:
        return LOADED_MODELS[cache_key]

    config = None
    for _, cfg in INDIVIDUAL_REACTION_CONFIGS.items():
        if cfg["key"] == reaction_key:
            config = cfg
            break

    if config is None:
        raise ValueError(f"Unknown individual reaction key: {reaction_key}")

    package_name = config["package"]

    model_defs_module = importlib.import_module(f"{package_name}.model_defs")
    features_module = importlib.import_module(f"{package_name}.features")
    config_module = importlib.import_module(f"{package_name}.config")

    register_classes_to_main(model_defs_module)

    model = torch.load(config["model_path"], map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()

    resources = {
        "model": model,
        "features": features_module,
        "meta": config_module.REACTION_META,
        "kind": "individual",
    }
    LOADED_MODELS[cache_key] = resources
    return resources

def load_fine_tuned_resources(category_label: str, reaction_key: str) -> Dict[str, Any]:
    cache_key = f"fine_tuned::{category_label}::{reaction_key}"
    if cache_key in LOADED_MODELS:
        return LOADED_MODELS[cache_key]

    if category_label not in FINE_TUNED_CONFIGS:
        raise ValueError(f"Unknown fine-tuned category: {category_label}")

    category_cfg = FINE_TUNED_CONFIGS[category_label]
    reaction_cfg = None

    for _, cfg in category_cfg["reactions"].items():
        if cfg["key"] == reaction_key:
            reaction_cfg = cfg
            break

    if reaction_cfg is None:
        raise ValueError(f"Unknown fine-tuned reaction key: {reaction_key}")

    package_name = reaction_cfg["package"]

    features_module = importlib.import_module(f"{package_name}.features")
    config_module = importlib.import_module(f"{package_name}.config")

    register_fine_tuned_shared_classes()

    model = torch.load(reaction_cfg["model_path"], map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()

    resources = {
        "model": model,
        "features": features_module,
        "meta": config_module.REACTION_META,
        "kind": "fine_tuned",
        "category": category_label,
    }
    LOADED_MODELS[cache_key] = resources
    return resources

def get_resources(model_group: str, reaction_key: str, fine_tuned_category: str = None):
    if model_group == "individual":
        return load_individual_resources(reaction_key)
    elif model_group == "fine_tuned":
        if not fine_tuned_category:
            raise ValueError("fine_tuned_category is required for fine-tuned models")
        return load_fine_tuned_resources(fine_tuned_category, reaction_key)
    else:
        raise ValueError(f"Unknown model_group: {model_group}")
    
def get_reaction_options_for_group(model_group: str, fine_tuned_category: str = None):
    if model_group == "individual":
        return [
            {"label": label, "key": cfg["key"]}
            for label, cfg in INDIVIDUAL_REACTION_CONFIGS.items()
        ]
    elif model_group == "fine_tuned":
        if not fine_tuned_category or fine_tuned_category not in FINE_TUNED_CONFIGS:
            return []
        return [
            {"label": label, "key": cfg["key"]}
            for label, cfg in FINE_TUNED_CONFIGS[fine_tuned_category]["reactions"].items()
        ]
    return []

# --------------------------------------------------
# Prediction runners for predefined reactions
# --------------------------------------------------
def run_prediction(model_group, model, features, meta, form_inputs):
    if model_group == "individual":
        return run_individual_prediction(model, features, meta, form_inputs)
    elif model_group == "fine_tuned":
        return run_fine_tuned_prediction(model, features, meta, form_inputs)
    else:
        raise ValueError(f"Unknown model_group: {model_group}")
    
def run_individual_prediction(model, features, meta, form_inputs):
    prediction_mode = meta["prediction_mode"]

    if prediction_mode == "standard_4graph":
        guide = features.clean_seq(form_inputs["guide"])
        target = features.clean_seq(form_inputs["target"])

        target = DNA_reverse_complement(target)

        features.validate_seq(guide, "guide")
        features.validate_seq(target, "target")

        x1 = features.build_sequence_features(guide, target).to(DEVICE)
        x2 = features.build_structure_features(guide, target).to(DEVICE)
        x3 = features.build_energy_features(guide, target).to(DEVICE)

        guide_data = features.build_guide_graph(guide)
        target_bh_data = features.build_target_bh_graph(target)
        duplex_data = features.build_duplex_graph(guide, target)
        target_ah_data = features.build_target_ah_graph(target)

        batch_guide = Batch.from_data_list([guide_data]).to(DEVICE)
        batch_target_bh = Batch.from_data_list([target_bh_data]).to(DEVICE)
        batch_duplex = Batch.from_data_list([duplex_data]).to(DEVICE)
        batch_target_ah = Batch.from_data_list([target_ah_data]).to(DEVICE)

        with torch.no_grad():
            pred = model(batch_guide, batch_target_bh, batch_duplex, batch_target_ah, x1, x2, x3)

        return float(pred.squeeze().cpu().item())

    elif prediction_mode == "cas13d_3graph":
        guide = features.clean_seq(form_inputs["guide"])
        target = features.clean_seq(form_inputs["target"])

        target = DNA_reverse_complement(target)

        features.validate_seq(guide, "guide")
        features.validate_seq(target, "target")

        x1 = features.build_sequence_features(guide, target).to(DEVICE)
        x2 = features.build_structure_features(guide, target).to(DEVICE)
        x3 = features.build_energy_features(guide, target).to(DEVICE)

        guide_data = features.build_guide_graph(guide)
        target_bh_data = features.build_target_bh_graph(target)
        duplex_data = features.build_duplex_graph(guide, target)

        batch_guide = Batch.from_data_list([guide_data]).to(DEVICE)
        batch_target_bh = Batch.from_data_list([target_bh_data]).to(DEVICE)
        batch_duplex = Batch.from_data_list([duplex_data]).to(DEVICE)

        with torch.no_grad():
            pred = model(batch_guide, batch_target_bh, batch_duplex, x1, x2, x3)

        return float(pred.squeeze().cpu().item())

    elif prediction_mode == "pe2_6graph":
        guide = features.clean_seq(form_inputs["guide"])
        target = features.clean_seq(form_inputs["target"])
        template = features.clean_seq(form_inputs["template"])

        target = DNA_reverse_complement(target)
        template = DNA_to_RNA(template)
        template = RNA_reverse_complement(template)

        features.validate_seq(guide, "guide")
        features.validate_seq(target, "target")
        features.validate_seq(template, "template")

        x1 = features.build_sequence_features(guide, target, template).to(DEVICE)
        x2 = features.build_structure_features(guide, target, template).to(DEVICE)
        x3 = features.build_energy_features(guide, target, template).to(DEVICE)

        guide_data = features.build_guide_graph(guide, template)
        target_bh_data = features.build_target_bh_graph(target)
        duplex_data = features.build_duplex_graph(guide, target, template)
        target_ah_data = features.build_target_ah_graph(target)
        triplex_bc_data = features.build_triplex_bc_graph(guide, target, template)
        triplex_ac_data = features.build_triplex_ac_graph(guide, target, template)

        batch_guide = Batch.from_data_list([guide_data]).to(DEVICE)
        batch_target_bh = Batch.from_data_list([target_bh_data]).to(DEVICE)
        batch_duplex = Batch.from_data_list([duplex_data]).to(DEVICE)
        batch_target_ah = Batch.from_data_list([target_ah_data]).to(DEVICE)
        batch_triplex_bc = Batch.from_data_list([triplex_bc_data]).to(DEVICE)
        batch_triplex_ac = Batch.from_data_list([triplex_ac_data]).to(DEVICE)

        with torch.no_grad():
            pred = model(
                batch_guide,
                batch_target_bh,
                batch_duplex,
                batch_target_ah,
                batch_triplex_bc,
                batch_triplex_ac,
                x1,
                x2,
                x3,
            )

        return float(pred.squeeze().cpu().item())

    else:
        raise ValueError(f"Unsupported individual prediction_mode: {prediction_mode}")
    
def run_fine_tuned_prediction(model, features, meta, form_inputs):
    prediction_mode = meta["prediction_mode"]

    if prediction_mode == "reduced_3graph":
        guide = features.clean_seq(form_inputs["guide"])
        target = features.clean_seq(form_inputs["target"])

        target = DNA_reverse_complement(target)

        features.validate_seq(guide, "guide")
        features.validate_seq(target, "target")

        x1 = features.build_sequence_features(guide, target).to(DEVICE)
        x2 = features.build_structure_features(guide, target).to(DEVICE)
        x3 = features.build_energy_features(guide, target).to(DEVICE)

        guide_data = features.build_guide_graph(guide)
        target_bh_data = features.build_target_bh_graph(target)
        duplex_data = features.build_duplex_graph(guide, target)

        batch_guide = Batch.from_data_list([guide_data]).to(DEVICE)
        batch_target_bh = Batch.from_data_list([target_bh_data]).to(DEVICE)
        batch_duplex = Batch.from_data_list([duplex_data]).to(DEVICE)

        with torch.no_grad():
            pred = model(batch_guide, batch_target_bh, batch_duplex, x1, x2, x3)

        return float(pred.squeeze().cpu().item())

    else:
        raise ValueError(f"Unsupported fine-tuned prediction_mode: {prediction_mode}")
    
# --------------------------------------------------
# Custom fine-tuning subsystem
# --------------------------------------------------
def load_custom_finetune_adapter(mode: str):
    if mode == "Cas9":
        return importlib.import_module("custom_finetune.cas9")
    elif mode == "Cas12":
        return importlib.import_module("custom_finetune.cas12")
    elif mode == "Cas13":
        return importlib.import_module("custom_finetune.cas13")
    else:
        raise ValueError(f"Unsupported custom fine-tuning mode: {mode}")
    
def read_user_xlsx(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    if df.shape[1] < 3:
        raise ValueError("The Excel file must contain at least 3 columns: guide, target, activity.")
    df = df.iloc[:, :3].copy()
    df.columns = ["guide", "target", "activity"]
    return df

def validate_uploaded_dataframe(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    df["guide"] = df["guide"].astype(str).str.strip().str.upper()
    df["target"] = df["target"].astype(str).str.strip().str.upper()
    df["activity"] = df["activity"].astype(float)

    if (df["guide"].str.len() > 30).any():
        raise ValueError("Guide sequence length must be at most 30 nt.")
    if (df["target"].str.len() > 30).any():
        raise ValueError("Target sequence length must be at most 30 nt.")
    if ((df["activity"] < 0) | (df["activity"] > 1)).any():
        raise ValueError("Activity values must be between 0 and 1.")

    return df

class UserHybridDataset(Dataset):
    def __init__(self, guide_graphs, target_bh_graphs, duplex_graphs, X1, X2, X3, y):
        self.guide_graphs = guide_graphs
        self.target_bh_graphs = target_bh_graphs
        self.duplex_graphs = duplex_graphs
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        gd = self.guide_graphs[idx]
        td = self.target_bh_graphs[idx]
        dd = self.duplex_graphs[idx]
        x1 = torch.tensor(self.X1[idx], dtype=torch.float32)
        x2 = torch.tensor(self.X2[idx], dtype=torch.float32)
        x3 = torch.tensor(self.X3[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return gd, td, dd, x1, x2, x3, y
    
def hybrid_collate_user(batch):
    c_list, t_list, d_list, x1_list, x2_list, x3_list, y_list = zip(*batch)

    batch_c = Batch.from_data_list(list(c_list))
    batch_t = Batch.from_data_list(list(t_list))
    batch_d = Batch.from_data_list(list(d_list))

    x1 = torch.stack(list(x1_list), dim=0)
    x2 = torch.stack(list(x2_list), dim=0)
    x3 = torch.stack(list(x3_list), dim=0)
    y = torch.stack(list(y_list), dim=0).view(-1)

    return batch_c, batch_t, batch_d, x1, x2, x3, y

def build_ft_loaders_from_subsets(
    user_dataset,
    train_idx,
    val_idx,
    batch_size,
    pin=False,
    repeat_k: int = 100,
    sampler_mode: str = "weighted",
    sampler_power: float = 1.2,
):
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)

    if repeat_k > 1:
        train_idx_rep = np.tile(train_idx, repeat_k)
    else:
        train_idx_rep = train_idx

    train_set = Subset(user_dataset, train_idx_rep.tolist())
    val_set = Subset(user_dataset, val_idx.tolist())

    if sampler_mode == "weighted":
        y_train = np.array([float(user_dataset.y[i]) for i in train_idx_rep], dtype=np.float64)

        med = np.median(y_train)
        spread = np.median(np.abs(y_train - med)) + 1e-8
        base_w = 1.0 + np.abs(y_train - med) / spread

        if sampler_power != 1.0:
            base_w = np.power(base_w, sampler_power)

        sampler = WeightedRandomSampler(
            weights=base_w.astype(np.float64),
            num_samples=len(base_w),
            replacement=True,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=hybrid_collate_user,
            num_workers=0,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=hybrid_collate_user,
            num_workers=0,
            pin_memory=pin,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=hybrid_collate_user,
        num_workers=0,
        pin_memory=pin,
    )

    return train_loader, val_loader

def make_loss(loss_name: str):
    if loss_name.lower() == "mse":
        return nn.MSELoss()
    elif loss_name.lower() == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    else:
        raise ValueError(f"Unknown loss '{loss_name}'")
    
def evaluate_mse(model, data_loader, device="cpu", loss_fn=None):
    loss_fn = loss_fn or nn.MSELoss()
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for c, t, d, x1, x2, x3, y in data_loader:
            c, t, d = c.to(device), t.to(device), d.to(device)
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            pred = model(c, t, d, x1, x2, x3)
            total_loss += float(loss_fn(pred, y).item())
            n_batches += 1

    if n_batches == 0:
        raise ValueError("Evaluation data loader is empty.")

    return total_loss / n_batches
    
def fine_tune_on_real(
    model,
    train_loader,
    val_loader=None,
    epochs=5000,
    lr=5e-4,
    weight_decay=1e-5,
    patience=20,
    device="cpu",
    freeze_backbone=False,
    loss_fn=None,
):
    model = model.to(device)

    if freeze_backbone:
        for m in [model.cnn_branch1, model.cnn_branch2, model.gnn_branch, model.mlp_branch]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(model.hidden.parameters()) + list(model.out.parameters())
        opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = loss_fn or nn.MSELoss()
    best_model_state = None
    best_val = float("inf")
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0

        for c, t, d, x1, x2, x3, y in train_loader:
            c, t, d = c.to(device), t.to(device), d.to(device)
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            pred = model(c, t, d, x1, x2, x3)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss_epoch += float(loss.item())

        avg_train_loss = total_loss_epoch / max(1, len(train_loader))

        if val_loader is None:
            print(f"[Fine-tune] Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")
            continue

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for c, t, d, x1, x2, x3, y in val_loader:
                c, t, d = c.to(device), t.to(device), d.to(device)
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                pred = model(c, t, d, x1, x2, x3)
                total_val += loss_fn(pred, y).item()

        avg_val_loss = total_val / max(1, len(val_loader))
        print(f"[Fine-tune] Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_model_state = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model = best_model_state

    return model, best_val

# --------------------------------------------------
# Routes: predefined reaction UI
# --------------------------------------------------  
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
        },
    )

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    model_group = "individual"
    reaction_options = get_reaction_options_for_group(model_group)
    selected_reaction_key = reaction_options[0]["key"] if reaction_options else None

    resources = get_resources(model_group, selected_reaction_key) if selected_reaction_key else None
    input_fields = resources["meta"]["input_fields"] if resources else []

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "model_group": model_group,
            "reaction_options": reaction_options,
            "selected_reaction_key": selected_reaction_key,
            "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
            "selected_fine_tuned_category": None,
            "input_fields": input_fields,
            "relative_activity": None,
            "output_label": None,
            "error": None,
        },
    )

@app.get("/custom-finetune", response_class=HTMLResponse)
def custom_finetune_page(request: Request):
    return templates.TemplateResponse(
        "custom_finetune.html",
        {
            "request": request,
            "custom_ft_success": False,
            "custom_ft_error": None,
            "custom_prediction": None,
        },
    )

@app.post("/select-model-group", response_class=HTMLResponse)
def select_model_group(
    request: Request,
    model_group: str = Form(...),
    fine_tuned_category: str = Form(None),
):
    if model_group == "individual":
        reaction_options = get_reaction_options_for_group("individual")
        selected_reaction_key = reaction_options[0]["key"] if reaction_options else None
        resources = get_resources("individual", selected_reaction_key) if selected_reaction_key else None
        input_fields = resources["meta"]["input_fields"] if resources else []

        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "model_group": "individual",
                "reaction_options": reaction_options,
                "selected_reaction_key": selected_reaction_key,
                "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
                "selected_fine_tuned_category": None,
                "input_fields": input_fields,
            },
        )

    if not fine_tuned_category:
        fine_tuned_category = list(FINE_TUNED_CONFIGS.keys())[0]

    reaction_options = get_reaction_options_for_group("fine_tuned", fine_tuned_category)
    selected_reaction_key = reaction_options[0]["key"] if reaction_options else None
    resources = get_resources("fine_tuned", selected_reaction_key, fine_tuned_category) if selected_reaction_key else None
    input_fields = resources["meta"]["input_fields"] if resources else []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_group": "fine_tuned",
            "reaction_options": reaction_options,
            "selected_reaction_key": selected_reaction_key,
            "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
            "selected_fine_tuned_category": fine_tuned_category,
            "input_fields": input_fields,
        },
    )

@app.post("/select-fine-tuned-category", response_class=HTMLResponse)
def select_fine_tuned_category(
    request: Request,
    model_group: str = Form(...),
    fine_tuned_category: str = Form(...),
):
    reaction_options = get_reaction_options_for_group("fine_tuned", fine_tuned_category)
    selected_reaction_key = reaction_options[0]["key"] if reaction_options else None
    resources = get_resources("fine_tuned", selected_reaction_key, fine_tuned_category) if selected_reaction_key else None
    input_fields = resources["meta"]["input_fields"] if resources else []

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "model_group": model_group,
            "reaction_options": reaction_options,
            "selected_reaction_key": selected_reaction_key,
            "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
            "selected_fine_tuned_category": fine_tuned_category,
            "input_fields": input_fields,
        },
    )

@app.post("/select-reaction", response_class=HTMLResponse)
def select_reaction(
    request: Request,
    model_group: str = Form(...),
    reaction_key: str = Form(...),
    fine_tuned_category: str = Form(None),
):
    resources = get_resources(model_group, reaction_key, fine_tuned_category)
    input_fields = resources["meta"]["input_fields"]

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "model_group": model_group,
            "reaction_options": get_reaction_options_for_group(model_group, fine_tuned_category),
            "selected_reaction_key": reaction_key,
            "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
            "selected_fine_tuned_category": fine_tuned_category,
            "input_fields": input_fields,
        },
    )

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()
    form_dict = dict(form)

    model_group = form_dict["model_group"]
    reaction_key = form_dict["reaction_key"]
    fine_tuned_category = form_dict.get("fine_tuned_category")

    try:
        resources = get_resources(model_group, reaction_key, fine_tuned_category)
        value = run_prediction(
            model_group,
            resources["model"],
            resources["features"],
            resources["meta"],
            form_dict,
        )
        display_value = transform_output_value(value, resources["meta"])
        output_label = resources["meta"].get("output_label", "Relative activity")

        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "model_group": model_group,
                "reaction_options": get_reaction_options_for_group(model_group, fine_tuned_category),
                "selected_reaction_key": reaction_key,
                "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
                "selected_fine_tuned_category": fine_tuned_category,
                "input_fields": resources["meta"]["input_fields"],
                "form_values": form_dict,
                "relative_activity": f"{display_value:.6f}",
                "output_label": output_label,
            },
        )
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_group": model_group,
                "reaction_options": get_reaction_options_for_group(model_group, fine_tuned_category),
                "selected_reaction_key": reaction_key,
                "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
                "selected_fine_tuned_category": fine_tuned_category,
                "input_fields": resources["meta"]["input_fields"] if "resources" in locals() else [],
                "form_values": form_dict,
                "error": str(e),
            },
        )
    
# --------------------------------------------------
# Routes: custom fine-tuning
# --------------------------------------------------
@app.post("/fine-tune-custom", response_class=HTMLResponse)
async def fine_tune_custom(request: Request):
    form = await request.form()

    finetune_mode = str(form.get("custom_mode", "")).strip()
    reaction_name = str(form.get("custom_reaction_name", "")).strip()
    scaffold_seq = str(form.get("custom_scaffold_seq", "")).strip()
    spacer_len_raw = str(form.get("custom_spacer_len", "")).strip()
    repeat_k_raw = str(form.get("custom_repeat_k", "")).strip()
    n_trials_raw = str(form.get("custom_n_trials", "")).strip()
    dataset_file = form.get("custom_dataset_file")

    try:
        if finetune_mode not in {"Cas9", "Cas12", "Cas13"}:
            raise ValueError("Please select a valid custom fine-tuning mode.")

        if not reaction_name:
            raise ValueError("Please provide a new reaction name.")

        if not scaffold_seq:
            raise ValueError("Please provide a gRNA scaffold sequence.")

        if not spacer_len_raw.isdigit():
            raise ValueError("Spacer length must be a positive integer.")
        
        if repeat_k_raw not in {"1", "10", "100"}:
            raise ValueError("Repeat_k must be one of 1, 10, or 100.") 
        repeat_k = int(repeat_k_raw)

        if not n_trials_raw.isdigit():
            raise ValueError("Number of trials must be a positive integer.")

        n_trials = int(n_trials_raw)
        if n_trials <= 0 or n_trials > 200:
            raise ValueError("Number of trials must be between 1 and 200.")

        spacer_len = int(spacer_len_raw)
        if spacer_len <= 0 or spacer_len > 30:
            raise ValueError("Spacer length must be between 1 and 30.")

        if dataset_file is None or not hasattr(dataset_file, "filename") or not dataset_file.filename:
            raise ValueError("Please upload a dataset file.")

        if not dataset_file.filename.lower().endswith(".xlsx"):
            raise ValueError("Dataset file must be an .xlsx Excel file.")

        adapter = load_custom_finetune_adapter(finetune_mode)

        job_id = str(uuid.uuid4())[:8]
        reaction_slug = slugify(reaction_name)
        work_dir = os.path.join(USER_MODEL_DIR, f"{reaction_slug}_{job_id}")
        os.makedirs(work_dir, exist_ok=True)

        uploaded_xlsx_path = os.path.join(work_dir, dataset_file.filename)
        with open(uploaded_xlsx_path, "wb") as f:
            f.write(await dataset_file.read())

        df = read_user_xlsx(uploaded_xlsx_path)
        df = validate_uploaded_dataframe(df)

        guide_graphs, target_bh_graphs, duplex_graphs, X1, X2, X3, y, params = adapter.build_training_components_from_dataframe(
            df,
            scaffold_seq=scaffold_seq,
            spacer_len=spacer_len,
        )

        user_dataset = UserHybridDataset(
            guide_graphs=guide_graphs,
            target_bh_graphs=target_bh_graphs,
            duplex_graphs=duplex_graphs,
            X1=X1,
            X2=X2,
            X3=X3,
            y=y,
        )

        n_samples = len(user_dataset)
        if n_samples < 5:
            raise ValueError("At least 5 samples are required for train/validation/unseen splitting.")

        all_indices = np.arange(n_samples)

        # 20% unseen, 80% seen
        seen_idx, unseen_idx = train_test_split(
            all_indices,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )

        # from seen: 80% train, 20% val
        train_idx, val_idx = train_test_split(
            seen_idx,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )

        pin = (DEVICE == "cuda")
        mse_loss = make_loss("mse")

        unseen_set = Subset(user_dataset, unseen_idx.tolist())
        unseen_loader = DataLoader(
            unseen_set,
            batch_size=min(512, max(1, len(unseen_set))),
            shuffle=False,
            collate_fn=hybrid_collate_user,
            num_workers=0,
            pin_memory=pin,
        )

        register_fine_tuned_shared_classes()

        model_path = os.path.join(work_dir, "fine_tuned_model.pt")
        best_trial_info = {}

        def objective(trial):
            model = build_custom_model_from_trial(trial).to(DEVICE)

            ft_lr = trial.suggest_float("ft_lr", 1e-6, 1e-4, log=True)
            ft_weight_decay = trial.suggest_float("ft_weight_decay", 1e-6, 1e-3, log=True)
            ft_batch_size = trial.suggest_categorical("ft_batch_size", [128, 256, 512])
            freeze_backbone = trial.suggest_categorical("freeze_backbone", [False])
            ft_loss_name = "mse"
            ft_epochs = 5000
            ft_patience = 20

            sampler_mode = trial.suggest_categorical("ft_sampler_mode", ["weighted"])
            sampler_power = trial.suggest_float("ft_sampler_power", 1.0, 1.5)

            train_loader, val_loader = build_ft_loaders_from_subsets(
                user_dataset=user_dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                batch_size=ft_batch_size,
                pin=(DEVICE == "cuda"),
                repeat_k=repeat_k,
                sampler_mode=sampler_mode,
                sampler_power=sampler_power,
            )

            model, val_mse = fine_tune_on_real(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=ft_epochs,
                lr=ft_lr,
                weight_decay=ft_weight_decay,
                patience=ft_patience,
                device=DEVICE,
                freeze_backbone=freeze_backbone,
                loss_fn=make_loss(ft_loss_name),
            )

            if "best_val" not in best_trial_info or val_mse < best_trial_info["best_val"]:
                best_trial_info["best_val"] = float(val_mse)
                best_trial_info["best_model"] = copy.deepcopy(model).to("cpu")
                best_trial_info["params"] = trial.params.copy()

            return val_mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_model = best_trial_info["best_model"].to(DEVICE)
        unseen_mse = evaluate_mse(
            model=best_model,
            data_loader=unseen_loader,
            device=DEVICE,
            loss_fn=mse_loss,
        )

        torch.save(best_model.to("cpu"), model_path)
        best_model = best_model.to(DEVICE)

        best_val_mse = float(best_trial_info["best_val"])
        best_params = best_trial_info["params"]

        metadata = {
            "reaction_name": reaction_name,
            "mode": finetune_mode,
            "scaffold_seq": scaffold_seq,
            "spacer_len": spacer_len,
            "repeat_k": repeat_k,
            "n_trials": n_trials,
            "n_samples": int(len(df)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_unseen": int(len(unseen_idx)),
            "val_mse": float(best_val_mse),
            "unseen_mse": float(unseen_mse),
            "best_trial_params": best_params,
            "model_path": model_path,
        }
        with open(os.path.join(work_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return templates.TemplateResponse(
            "custom_finetune.html",
            {
                "request": request,
                "custom_ft_success": True,
                "custom_ft_error": None,
                "custom_reaction_name": reaction_name,
                "custom_mode": finetune_mode,
                "custom_scaffold_seq": scaffold_seq,
                "custom_spacer_len": spacer_len,
                "custom_repeat_k": repeat_k,
                "custom_n_trials": n_trials,
                "custom_n_samples": len(df),
                "custom_n_train": len(train_idx),
                "custom_n_val": len(val_idx),
                "custom_n_unseen": len(unseen_idx),
                "custom_val_mse": f"{best_val_mse:.6f}",
                "custom_unseen_mse": f"{unseen_mse:.6f}",
                "custom_model_path": model_path,
                "download_model_url": f"/download-custom-model/{reaction_slug}_{job_id}",
            },
        )

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "custom_ft_success": False,
                "custom_ft_error": str(e),
                "custom_mode": finetune_mode,
                "custom_reaction_name": reaction_name,
                "custom_scaffold_seq": scaffold_seq,
                "custom_spacer_len": spacer_len_raw,
                "custom_repeat_k": repeat_k_raw,
                "custom_n_trials": n_trials_raw,
            },
        )
    
@app.get("/download-custom-model/{job_name}")
def download_custom_model(job_name: str):
    model_path = os.path.join(USER_MODEL_DIR, job_name, "fine_tuned_model.pt")
    if not os.path.exists(model_path):
        return {"error": "Model file not found."}
    return FileResponse(
        model_path,
        filename="fine_tuned_model.pt",
        media_type="application/octet-stream",
    )

@app.get("/download-dataset-template")
def download_dataset_template():
    template_path = os.path.join("downloads", "dataset_template.xlsx")

    if not os.path.exists(template_path):
        return {"error": "Dataset template file not found."}

    return FileResponse(
        template_path,
        filename="dataset_template.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@app.post("/predict-custom-model", response_class=HTMLResponse)
async def predict_custom_model(request: Request):
    form = await request.form()

    custom_model_path = str(form.get("custom_model_path", "")).strip()
    custom_mode = str(form.get("custom_mode", "")).strip()
    custom_scaffold_seq = str(form.get("custom_scaffold_seq", "")).strip()
    custom_spacer_len_raw = str(form.get("custom_spacer_len", "")).strip()
    custom_guide = str(form.get("custom_guide", "")).strip()
    custom_target = str(form.get("custom_target", "")).strip()

    try:
        if not custom_model_path:
            raise ValueError("Custom model path is missing.")

        if not os.path.exists(custom_model_path):
            raise ValueError("Custom fine-tuned model file does not exist.")

        if custom_mode not in {"Cas9", "Cas12", "Cas13"}:
            raise ValueError("Invalid custom model mode.")

        if not custom_spacer_len_raw.isdigit():
            raise ValueError("Spacer length must be a positive integer.")

        spacer_len = int(custom_spacer_len_raw)
        adapter = load_custom_finetune_adapter(custom_mode)

        register_fine_tuned_shared_classes()
        model = torch.load(custom_model_path, map_location=DEVICE, weights_only=False)
        model.to(DEVICE)
        model.eval()

        value = adapter.predict_single(
            model=model,
            guide=custom_guide,
            target=custom_target,
            scaffold_seq=custom_scaffold_seq,
            spacer_len=spacer_len,
            device=DEVICE,
        )

        return templates.TemplateResponse(
            "custom_finetune.html",
            {
                "request": request,
                "custom_ft_success": True,
                "custom_ft_error": None,
                "custom_model_path": custom_model_path,
                "custom_mode": custom_mode,
                "custom_scaffold_seq": custom_scaffold_seq,
                "custom_spacer_len": spacer_len,
                "custom_prediction": f"{value:.6f}",
                "custom_output_label": "Relative activity",
                "custom_guide": custom_guide,
                "custom_target": custom_target,
            },
        )

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "custom_ft_success": True,
                "custom_ft_error": str(e),
                "custom_model_path": custom_model_path,
                "custom_mode": custom_mode,
                "custom_scaffold_seq": custom_scaffold_seq,
                "custom_spacer_len": custom_spacer_len_raw,
                "custom_guide": custom_guide,
                "custom_target": custom_target,
            },
        )
