# reactions/RfxCas13d-mismatch_Wessels_2024_NB/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 23-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 23-nt sequence in RNA"},
    ],
    "prediction_mode": "cas13d_3graph",
}