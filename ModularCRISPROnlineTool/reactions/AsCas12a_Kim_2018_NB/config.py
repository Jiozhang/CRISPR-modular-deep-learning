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
         "instruction": "Please input a 50-nt sequence in DNA, in the form of 16-nt prefix + 20-nt target + 14-nt suffix"},
    ],
    "prediction_mode": "standard_4graph",
}