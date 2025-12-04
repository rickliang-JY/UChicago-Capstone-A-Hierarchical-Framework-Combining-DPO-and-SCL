```
notice: there are more negative data in the HER2 dataset, when we run the logistic regression, we use: "class_weight = 'balanced'", check the eval/-linear_probe.py in this folder.

!python use_supcon_head.py \
    --skl_path '/content/gdrive/MyDrive/Capstone Code/supcon testing/data/dpo_data_stratify_erbb2_embeddings_epoch40_lr0.001_beta0.5_20checkpoints.pkl' \
    --csv_path ../data/ERBB2_manifest.csv \
    --biomarker ERBB2 \
    --supcon_checkpoint  ./test/pretrain_ERBB2_bs32_lr0.001_temp0.02/checkpoint_epoch_20.pth
```
