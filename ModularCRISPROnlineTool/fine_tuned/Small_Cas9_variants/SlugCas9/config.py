# fine_tuned/Small_Cas9_variants/SlugCas9/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 21-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 21-nt sequence in DNA"},
    ],
    "prediction_mode": "reduced_3graph",
}