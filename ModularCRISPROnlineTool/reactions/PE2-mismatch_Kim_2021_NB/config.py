# reactions/PE2-mismatch_Kim_2021_NB/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 20-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 47-nt sequence in DNA, in the form of 23-nt prefix + 20-nt target + 4-nt suffix"},
        {"name": "template", 
         "label": "Template", 
         "placeholder": "Enter template sequence",
         "instruction": "Please input a maximum 37-nt sequence in RNA"},
    ],
    "prediction_mode": "pe2_6graph",
}