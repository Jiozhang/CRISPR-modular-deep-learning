import traceback
import importlib
import __main__
from typing import Dict, Any

import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torch_geometric.data import Batch

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

def register_fine_tuned_shared_classes():
    __main__.CNNBranch1 = CNNBranch1
    __main__.CNNBranch2 = CNNBranch2
    __main__.MLPBranch120 = MLPBranch120
    __main__.GNNBranch = GNNBranch
    __main__.CNN_GNN_MLP_Fusion = CNN_GNN_MLP_Fusion

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

def register_classes_to_main(module):
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            setattr(__main__, name, obj)

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

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    model_group = "individual"
    reaction_options = get_reaction_options_for_group(model_group)
    selected_reaction_key = reaction_options[0]["key"] if reaction_options else None

    resources = get_resources(model_group, selected_reaction_key) if selected_reaction_key else None
    input_fields = resources["meta"]["input_fields"] if resources else []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_group": model_group,
            "reaction_options": reaction_options,
            "selected_reaction_key": selected_reaction_key,
            "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
            "selected_fine_tuned_category": None,
            "input_fields": input_fields,
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
            "index.html",
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
        "index.html",
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
        "index.html",
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

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_group": model_group,
                "reaction_options": get_reaction_options_for_group(model_group, fine_tuned_category),
                "selected_reaction_key": reaction_key,
                "fine_tuned_categories": list(FINE_TUNED_CONFIGS.keys()),
                "selected_fine_tuned_category": fine_tuned_category,
                "input_fields": resources["meta"]["input_fields"],
                "form_values": form_dict,
                "ratio": f"{value:.6f}",
                "percent": f"{value * 100:.2f}",
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





