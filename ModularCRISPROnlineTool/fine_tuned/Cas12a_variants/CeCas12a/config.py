# fine_tuned/Cas12a_variants/CeCas12a/config.py

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
}