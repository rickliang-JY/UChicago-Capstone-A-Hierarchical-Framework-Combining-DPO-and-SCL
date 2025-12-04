```
python main_slide_level_improved.py   --skl_path='../data/pgr_dpo_data_stratify_embeddings_epoch40_lr0.001_beta0.5_best_model_20_epochs.pkl'   --csv_path='../data/PGR_manifest.csv'   --biomarker 'PGR'   --mode pretrain   --batch_size 64   --hidden_dim 256   --feat_dim 64   --num_layers 2   --learning_rate 0.002   --temperature 0.01   --scheduler cosine   --epochs 200   --save_freq 10   --experiment_name pgr_dpo_supcon_optimized_afterDPO_batch64_hd256_feat64_layer2_lr0.002_t0.01_cosine_epochs200
python use_supcon_head.py     --skl_path='../data/pgr_dpo_data_stratify_embeddings_epoch40_lr0.001_beta0.5_best_model_20_epochs.pkl'    --csv_path='../data/PGR_manifest.csv'       --biomarker PGR    --hidden_dim  256   --feat_dim 64     --num_layer 2     --batch_size 64     --supcon_checkpoint ./checkpoints/pgr_dpo_supcon_optimized_afterDPO_batch64_hd256_feat64_layer2_lr0.002_t0.01_cosine_epochs200/checkpoint_epoch_10.pth
/acc                : 0.8226
/bacc               : 0.7457
/kappa              : 0.5443
/nw_kappa           : 0.5443
/weighted_f1        : 0.8099
/loss               : 0.4758
/auroc              : 0.8192
/auprc              : 0.8926

EmbeddingRefinementModel 

from dataset_simple_random_split import (
    get_dataloaders, SimpleSupConLoss, get_class_weights
)
```
