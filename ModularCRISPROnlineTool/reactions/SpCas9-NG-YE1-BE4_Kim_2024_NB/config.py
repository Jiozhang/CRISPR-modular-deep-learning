# reactions/SpCas9-NG-YE1-BE4_Kim_2024_NB/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 20-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 30-nt sequence in DNA, in the form of 6-nt prefix + 20-nt target + 4-nt suffix"},
    ],
    "prediction_mode": "standard_4graph",
}