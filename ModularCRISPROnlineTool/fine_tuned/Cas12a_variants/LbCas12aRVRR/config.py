# fine_tuned/Cas12a_variants/LbCas12aRVRR/config.py

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
    "prediction_mode": "reduced_3graph",
    "output_label": "Relative activity",
    "output_transform": {
        "type": "identity"
    }
}