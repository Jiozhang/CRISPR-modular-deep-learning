# reactions/CHANGE-seq-mismatch_Lazzarotto_2020_NB/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 20-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 20-nt sequence in DNA"},
    ],
    "prediction_mode": "standard_4graph",
    "output_label": "Relative activity",
    "output_transform": {
        "type": "divide",
        "value": 4.201,
        "clip_0_1": True
    }
}