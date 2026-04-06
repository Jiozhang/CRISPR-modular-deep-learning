# reactions/SpCas9-mismatch_Doench_2016_NB/config.py

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
        "type": "minmax",
        "min": -5.38,
        "max": 3.91,
        "clip_0_1": True
    }
}