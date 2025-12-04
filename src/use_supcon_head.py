#!/usr/bin/env python3
"""

"""
import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

from models_pathology import SupConEmbeddingModel

from dataset_simple_random_split import (
    get_dataloaders, SimpleSupConLoss, get_class_weights
)

from eval_linear_probe import train_and_evaluate_logistic_regression_with_val


def parse_args():
    parser = argparse.ArgumentParser('Use SupCon Head for Evaluation')
    
    # Data
    parser.add_argument('--skl_path', type=str, required=True,
                        help='Path to embeddings pickle file')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to clinical CSV file')
    parser.add_argument('--biomarker', type=str, required=True,
                        choices=['ESR1', 'PR', 'ERBB2', 'PGR', 'MSI'],
                        help='Biomarker to predict')
    
    # SupCon model
    parser.add_argument('--supcon_checkpoint', type=str, required=True,
                        help='Path to trained SupCon model checkpoint')
    parser.add_argument('--input_dim', type=int, default=768,
                        help='Input embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='Output feature dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    
    # Evaluation
    parser.add_argument('--compare_baseline', action='store_true',
                        help='Also evaluate baseline (original embeddings)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id')
    
    return parser.parse_args()


def load_supcon_model(checkpoint_path, input_dim, hidden_dim, feat_dim, 
                      num_layers, device):

    print(f"\nLoading SupCon model from: {checkpoint_path}")
    
    model = SupConEmbeddingModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        feat_dim=feat_dim,
        num_layers=num_layers
    )
    
    # PyTorch 2.6+ compatibility: need weights_only=False for checkpoints with args
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    
    print(" SupCon model loaded successfully")
    
    return model


def extract_embeddings(dataloader, supcon_model, use_head, device):
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in tqdm(dataloader, desc='Extracting'):
            batch_embeddings = batch_embeddings.to(device)
            
            if use_head:
                # é€šè¿‡projection headèŽ·å–refined embeddings
                refined = supcon_model(batch_embeddings)
                all_embeddings.append(refined.cpu().numpy())
            else:
                # use original embeddings
                all_embeddings.append(batch_embeddings.cpu().numpy())
            
            all_labels.append(batch_labels.numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Using SupCon Projection Head for Evaluation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Biomarker: {args.biomarker}")
    print(f"SupCon checkpoint: {args.supcon_checkpoint}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    dataloaders, datasets = get_dataloaders(
        skl_path=args.skl_path,
        csv_path=args.csv_path,
        biomarker=args.biomarker,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode='linear_eval'  # No augmentation
    )
    
    # Load SupCon model
    print("\n" + "="*70)
    print("Loading SupCon Model")
    print("="*70)
    supcon_model = load_supcon_model(
        checkpoint_path=args.supcon_checkpoint,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
        num_layers=args.num_layers,
        device=device
    )
    
    # Extract refined embeddings (using projection head)
    print("\n" + "="*70)
    print("Extracting Refined Embeddings (with Projection Head)")
    print("="*70)
    
    train_X_refined, train_y = extract_embeddings(
        dataloaders['train'], supcon_model, use_head=True, device=device
    )
    val_X_refined, val_y = extract_embeddings(
        dataloaders['val'], supcon_model, use_head=True, device=device
    )
    test_X_refined, test_y = extract_embeddings(
        dataloaders['test'], supcon_model, use_head=True, device=device
    )
    
    print(f"\nRefined embeddings shape:")
    print(f"  Train: {train_X_refined.shape}")
    print(f"  Val:   {val_X_refined.shape}")
    print(f"  Test:  {test_X_refined.shape}")
    
    # Train and evaluate on refined embeddings
    print("\n" + "="*70)
    print("Training Logistic Regression on Refined Embeddings")
    print("="*70)
    
    metrics_refined, outputs_refined = train_and_evaluate_logistic_regression_with_val(
        train_data=train_X_refined,
        train_labels=train_y,
        val_data=val_X_refined,
        val_labels=val_y,
        test_data=test_X_refined,
        test_labels=test_y
    )
    
    # Print results
    print("\n" + "="*70)
    print(f"RESULTS - {args.biomarker} - WITH PROJECTION HEAD")
    print("="*70)
    for metric, value in metrics_refined.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # Save results
    results_path = os.path.join(
        args.output_dir,
        f'{args.biomarker}_supcon_head_results.pkl'
    )
    
    with open(results_path, 'wb') as f:
        pickle.dump({
            'metrics': metrics_refined,
            'outputs': outputs_refined,
            'args': vars(args)
        }, f)
    
    print(f"\n Results saved to: {results_path}")
    
    # Baseline comparison (optional)
    if args.compare_baseline:
        print("\n" + "="*70)
        print("Extracting Original Embeddings (Baseline)")
        print("="*70)
        
        train_X_baseline, _ = extract_embeddings(
            dataloaders['train'], supcon_model, use_head=False, device=device
        )
        val_X_baseline, _ = extract_embeddings(
            dataloaders['val'], supcon_model, use_head=False, device=device
        )
        test_X_baseline, _ = extract_embeddings(
            dataloaders['test'], supcon_model, use_head=False, device=device
        )
        
        print(f"\nOriginal embeddings shape:")
        print(f"  Train: {train_X_baseline.shape}")
        print(f"  Val:   {val_X_baseline.shape}")
        print(f"  Test:  {test_X_baseline.shape}")
        
        print("\n" + "="*70)
        print("Training Logistic Regression on Original Embeddings")
        print("="*70)
        
        metrics_baseline, outputs_baseline = train_and_evaluate_logistic_regression_with_val(
            train_data=train_X_baseline,
            train_labels=train_y,
            val_data=val_X_baseline,
            val_labels=val_y,
            test_data=test_X_baseline,
            test_labels=test_y
        )
        
        print("\n" + "="*70)
        print(f"RESULTS - {args.biomarker} - BASELINE (NO HEAD)")
        print("="*70)
        for metric, value in metrics_baseline.items():
            print(f"{metric:20s}: {value:.4f}")
        
        # Comparison
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        for metric in metrics_refined.keys():
            refined_val = metrics_refined[metric]
            baseline_val = metrics_baseline[metric]
            improvement = refined_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
            
            print(f"{metric:20s}:")
            print(f"  With Head:    {refined_val:.4f}")
            print(f"  Without Head: {baseline_val:.4f}")
            print(f"  Improvement:  {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        # Save baseline results
        baseline_path = os.path.join(
            args.output_dir,
            f'{args.biomarker}_baseline_results.pkl'
        )
        
        with open(baseline_path, 'wb') as f:
            pickle.dump({
                'metrics': metrics_baseline,
                'outputs': outputs_baseline,
                'args': vars(args)
            }, f)
        
        print(f"\n Baseline results saved to: {baseline_path}")



if __name__ == '__main__':
    main()