# WatchMyBirds-Classifier Models

This repository contains machine learning models trained for the [**WatchMyBirds**](https://github.com/arminfabritzek/WatchMyBirds) project, used for classifying bird species from images.

The models were trained using data derived from [**WatchMyBirds-Data**](https://github.com/arminfabritzek/WatchMyBirds-Data).

Dataset potentially used for training included **29** classes.
General dataset balancing parameters (if applicable, from basic settings):
- Minimum samples per class for inclusion: `400`
- Maximum samples per class (capped at): `470`
*(Note: Specific runs might use different parameters, check details below)*

---
## Trained Model Versions

Below are details for each available training run, typically sorted newest first. Models and associated artifacts are stored in the `models/` directory, organized by training timestamp.


### Model Version: `20250404_095937` (Type: `efficientnet_b1`)
**Confusion Matrix:**![](models/20250404_095937/confusion_matrix.png)
**Training Parameters (Table):**| Key | Value |
| --- | --- |
| adamW_weight_decay | 0.0007 |
| batch_size | 256 |
| criterion | `{"type": "CrossEntropyLoss", "label_smoothing": 0.1}` |
| early_stop_patience | 15 |
| full_finetune_lr | 0.0002 |
| image_size | 240 |
| initial_head_lr | 0.002 |
| model_name | efficientnet_b1 |
| num_classes | 29 |
| num_epochs_total | 200 |
| optimizer_type | AdamW |
| scheduler_type | SequentialLR (LinearWarmup + CosineAnnealing) |
| unfreeze_epoch | 10 |
| warmup_epochs | 5 |

**Training Parameters (Raw JSON):**```json
{
    "adamW_weight_decay": 0.0007,
    "batch_size": 256,
    "criterion": {
        "label_smoothing": 0.1,
        "type": "CrossEntropyLoss"
    },
    "early_stop_patience": 15,
    "full_finetune_lr": 0.0002,
    "image_size": 240,
    "initial_head_lr": 0.002,
    "model_name": "efficientnet_b1",
    "num_classes": 29,
    "num_epochs_total": 200,
    "optimizer_type": "AdamW",
    "scheduler_type": "SequentialLR (LinearWarmup + CosineAnnealing)",
    "unfreeze_epoch": 10,
    "warmup_epochs": 5
}
```
---


### Model Version: `20250328_061115` (Type: `efficientnet_b0`)
**Confusion Matrix:**![](models/20250328_061115/confusion_matrix.png)
**Training Parameters (Table):**| Key | Value |
| --- | --- |
| adamW_weight_decay | 0.0004 |
| batch_size | 256 |
| criterion | `{"type": "CrossEntropyLoss", "label_smoothing": 0.1}` |
| early_stop_patience | 15 |
| full_finetune_lr | 0.0001 |
| image_size | 224 |
| initial_head_lr | 0.001 |
| model_name | efficientnet_b0 |
| num_classes | 22 |
| num_epochs_total | 200 |
| optimizer_type | AdamW |
| scheduler_type | SequentialLR (LinearWarmup + CosineAnnealing) |
| unfreeze_epoch | 10 |
| warmup_epochs | 5 |

**Training Parameters (Raw JSON):**```json
{
    "adamW_weight_decay": 0.0004,
    "batch_size": 256,
    "criterion": {
        "label_smoothing": 0.1,
        "type": "CrossEntropyLoss"
    },
    "early_stop_patience": 15,
    "full_finetune_lr": 0.0001,
    "image_size": 224,
    "initial_head_lr": 0.001,
    "model_name": "efficientnet_b0",
    "num_classes": 22,
    "num_epochs_total": 200,
    "optimizer_type": "AdamW",
    "scheduler_type": "SequentialLR (LinearWarmup + CosineAnnealing)",
    "unfreeze_epoch": 10,
    "warmup_epochs": 5
}
```
---

---
## Detailed Dataset Statistics

The following statistics are based on the `dataset_reference.json` file provided during README generation.

# Dataset Statistics

## Train Split
| Species | Number of Images |
|---------|------------------|
| Aegithalos_caudatus | 470 |
| Carduelis_carduelis | 470 |
| Certhia_brachydactyla | 470 |
| Chloris_chloris | 470 |
| Coccothraustes_coccothraustes | 470 |
| Columba_palumbus | 470 |
| Cyanistes_caeruleus | 470 |
| Dendrocopos_major | 470 |
| Emberiza_citrinella | 470 |
| Erithacus_rubecula | 470 |
| Fringilla_coelebs | 470 |
| Fringilla_montifringilla | 470 |
| Garrulus_glandarius | 470 |
| Lophophanes_cristatus | 470 |
| Parus_major | 470 |
| Passer_domesticus | 470 |
| Periparus_ater | 470 |
| Phoenicurus_ochruros | 470 |
| Phylloscopus_collybita | 470 |
| Phylloscopus_trochilus | 470 |
| Pica_pica | 470 |
| Poecile_palustris | 470 |
| Pyrrhula_pyrrhula | 470 |
| Sitta_europaea | 470 |
| Spinus_spinus | 470 |
| Sylvia_atricapilla | 470 |
| Troglodytes_troglodytes | 470 |
| Turdus_merula | 470 |
| Turdus_philomelos | 470 |

**Total images in train: 13630**

## Val Split
| Species | Number of Images |
|---------|------------------|
| Aegithalos_caudatus | 89 |
| Carduelis_carduelis | 99 |
| Certhia_brachydactyla | 99 |
| Chloris_chloris | 114 |
| Coccothraustes_coccothraustes | 92 |
| Columba_palumbus | 91 |
| Cyanistes_caeruleus | 99 |
| Dendrocopos_major | 96 |
| Emberiza_citrinella | 89 |
| Erithacus_rubecula | 92 |
| Fringilla_coelebs | 95 |
| Fringilla_montifringilla | 98 |
| Garrulus_glandarius | 82 |
| Lophophanes_cristatus | 96 |
| Parus_major | 89 |
| Passer_domesticus | 99 |
| Periparus_ater | 90 |
| Phoenicurus_ochruros | 91 |
| Phylloscopus_collybita | 106 |
| Phylloscopus_trochilus | 92 |
| Pica_pica | 93 |
| Poecile_palustris | 94 |
| Pyrrhula_pyrrhula | 95 |
| Sitta_europaea | 86 |
| Spinus_spinus | 96 |
| Sylvia_atricapilla | 88 |
| Troglodytes_troglodytes | 99 |
| Turdus_merula | 97 |
| Turdus_philomelos | 87 |

**Total images in val: 2733**

## Test Split
| Species | Number of Images |
|---------|------------------|
| Aegithalos_caudatus | 11 |
| Carduelis_carduelis | 13 |
| Certhia_brachydactyla | 12 |
| Chloris_chloris | 15 |
| Coccothraustes_coccothraustes | 12 |
| Columba_palumbus | 13 |
| Cyanistes_caeruleus | 11 |
| Dendrocopos_major | 12 |
| Emberiza_citrinella | 12 |
| Erithacus_rubecula | 11 |
| Fringilla_coelebs | 11 |
| Fringilla_montifringilla | 14 |
| Garrulus_glandarius | 10 |
| Lophophanes_cristatus | 12 |
| Parus_major | 13 |
| Passer_domesticus | 11 |
| Periparus_ater | 11 |
| Phoenicurus_ochruros | 12 |
| Phylloscopus_collybita | 13 |
| Phylloscopus_trochilus | 12 |
| Pica_pica | 11 |
| Poecile_palustris | 12 |
| Pyrrhula_pyrrhula | 11 |
| Sitta_europaea | 11 |
| Spinus_spinus | 12 |
| Sylvia_atricapilla | 11 |
| Troglodytes_troglodytes | 12 |
| Turdus_merula | 11 |
| Turdus_philomelos | 11 |

**Total images in test: 343**

## Overall Dataset Summary
**Total images in dataset:** 16706

### Overall Species Counts
| Species | Total Number of Images |
|---------|------------------------|
| Aegithalos_caudatus | 570 |
| Carduelis_carduelis | 582 |
| Certhia_brachydactyla | 581 |
| Chloris_chloris | 599 |
| Coccothraustes_coccothraustes | 574 |
| Columba_palumbus | 574 |
| Cyanistes_caeruleus | 580 |
| Dendrocopos_major | 578 |
| Emberiza_citrinella | 571 |
| Erithacus_rubecula | 573 |
| Fringilla_coelebs | 576 |
| Fringilla_montifringilla | 582 |
| Garrulus_glandarius | 562 |
| Lophophanes_cristatus | 578 |
| Parus_major | 572 |
| Passer_domesticus | 580 |
| Periparus_ater | 571 |
| Phoenicurus_ochruros | 573 |
| Phylloscopus_collybita | 589 |
| Phylloscopus_trochilus | 574 |
| Pica_pica | 574 |
| Poecile_palustris | 576 |
| Pyrrhula_pyrrhula | 576 |
| Sitta_europaea | 567 |
| Spinus_spinus | 578 |
| Sylvia_atricapilla | 569 |
| Troglodytes_troglodytes | 581 |
| Turdus_merula | 578 |
| Turdus_philomelos | 568 |
