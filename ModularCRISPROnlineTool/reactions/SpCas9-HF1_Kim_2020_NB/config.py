# reactions/SpCas9-HF1_Kim_2020_NB/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 20-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 30-nt sequence in DNA, in the form of 4-nt prefix + 20-nt target + 6-nt suffix"},
    ],
    "prediction_mode": "standard_4graph",
    "output_label": "Relative activity",
    "output_transform": {
        "type": "identity"
    }
}