# fine_tuned/Base_editor_variants/SpCas9-NRCH-APOBEC-nCas9-Ung-3C/config.py

REACTION_META = {
    "input_fields": [
        {"name": "guide", 
         "label": "Guide", 
         "placeholder": "Enter guide sequence",
         "instruction": "Please input a 19-nt sequence in RNA"},
        {"name": "target", 
         "label": "Target", 
         "placeholder": "Enter target sequence",
         "instruction": "Please input a 19-nt sequence in DNA"},
    ],
    "prediction_mode": "reduced_3graph",
    "output_label": "Editing efficiency",
    "output_transform": {
        "type": "identity"
    }
}