# reactions/AsCas12a_Kim_2018_NB/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 20-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 50-nt sequence in DNA, in the form of 14-nt prefix + 20-nt target + 16-nt suffix"},
    ],
    "prediction_mode": "standard_4graph",
    "output_label": "Relative activity",
    "output_transform": {
        "type": "identity"
    }
}