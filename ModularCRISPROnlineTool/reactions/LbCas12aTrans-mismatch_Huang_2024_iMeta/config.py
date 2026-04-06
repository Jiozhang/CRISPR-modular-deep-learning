# reactions/LbCas12aTrans-mismatch_Huang_2024_iMeta/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 21-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 21-nt sequence in DNA"},
    ],
    "prediction_mode": "standard_4graph",
    "output_label": "Relative activity",
    "output_transform": {
        "type": "minmax",
        "min": -4.21,
        "max": -0.30,
        "clip_0_1": True
    }
}