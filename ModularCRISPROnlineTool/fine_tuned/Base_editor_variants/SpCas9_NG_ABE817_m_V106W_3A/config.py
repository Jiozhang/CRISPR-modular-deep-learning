# fine_tuned/Base_editor_variants/SpCas9-NG-ABE8.17-m+V106W-3A/config.py

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