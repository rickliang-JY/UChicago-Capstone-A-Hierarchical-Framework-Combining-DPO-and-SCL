"""
Improved training script with clear modes:
1. pretrain: Train SupCon projection head
2. classifier: Train classifier directly on embeddings (NO pretrain needed)
3. finetune: Load pretrained SupCon model, train classifier on refined embeddings
"""
import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)


from dataset_simple_random_split import (
    get_dataloaders, SimpleSupConLoss, get_class_weights
)
from models_pathology import (
    SupConEmbeddingModel, LinearClassifier, MLPClassifier
)


def parse_args():
    parser = argparse.ArgumentParser('TITAN Slide-Level Training')
    
    # Data parameters
    parser.add_argument('--skl_path', type=str, required=True,
                        help='Path to pickle/SKL file with slide embeddings')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV with clinical annotations')
    parser.add_argument('--biomarker', type=str, required=True,
                        choices=['ESR1', 'PR', 'ERBB2', 'PGR', 'MSI'],
                        help='Biomarker to predict')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=768,
                        help='TITAN embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for projection head')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='Feature dimension for contrastive space')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in projection head')
    
    # Training parameters
    parser.add_argument('--mode', type=str, default='classifier',
                        choices=['pretrain', 'classifier', 'finetune'],
                        help='Training mode: pretrain=SupCon only, classifier=direct classifier, finetune=SupCon+classifier')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    
    # Class imbalance handling
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss for classification')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    
    # For finetune mode
    parser.add_argument('--pretrained_supcon', type=str, default=None,
                        help='Path to pre-trained SupCon model (for finetune mode)')
    
    # Checkpoint and logging
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Checkpoint save frequency')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    
    args = parser.parse_args()
    
    # Set experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.mode}_{args.biomarker}_"
            f"bs{args.batch_size}_lr{args.learning_rate}"
        )
    
    # Create save directory
    args.save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, 
                                              weight=self.alpha,
                                              reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_contrastive(train_loader, model, criterion, optimizer, epoch, args):
    """One epoch of contrastive training"""
    model.train()
    
    losses = []
    batch_time = []
    
    for idx, (views, labels) in enumerate(train_loader):
        start_time = time.time()
        
        # Move to GPU
        view1, view2 = views[0].cuda(), views[1].cuda()
        labels = labels.cuda()
        
        # Forward pass for both views
        feat1 = model(view1)
        feat2 = model(view2)
        
        # Stack features: [bsz, n_views, feat_dim]
        features = torch.stack([feat1, feat2], dim=1)
        
        # Compute contrastive loss
        loss = criterion(features, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record
        losses.append(loss.item())
        batch_time.append(time.time() - start_time)
        
        # Print
        if (idx + 1) % args.print_freq == 0:
            print(f'Epoch [{epoch}][{idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} ({np.mean(losses):.4f}) '
                  f'Time: {batch_time[-1]:.3f}s')
    
    return np.mean(losses)


def train_classifier(train_loader, model, criterion, optimizer, epoch, args):
    """One epoch of classifier training"""
    model.train()
    
    losses = []
    all_preds = []
    all_labels = []
    
    for idx, (embeddings, labels) in enumerate(train_loader):
        # Move to GPU
        embeddings = embeddings.cuda()
        labels = labels.cuda()
        
        # Forward pass
        logits = model(embeddings)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Print
        if (idx + 1) % args.print_freq == 0:
            acc = accuracy_score(all_labels, all_preds)
            print(f'Epoch [{epoch}][{idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} ({np.mean(losses):.4f}) '
                  f'Acc: {acc:.4f}')
    
    train_acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), train_acc


def evaluate(data_loader, model, criterion, mode, args):
    """Evaluate model"""
    model.eval()
    
    losses = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.cuda()
            labels = labels.cuda()
            
            if mode == 'contrastive':
                # For contrastive mode, we don't have a classifier
                # Just return dummy values
                return 0.0, {}
            else:
                # Classification mode
                logits = model(embeddings)
                loss = criterion(logits, labels)
                
                losses.append(loss.item())
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
    
    # Compute metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()
    preds = torch.argmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    
    metrics = {}
    metrics['auroc'] = roc_auc_score(labels_np, probs)
    metrics['auprc'] = average_precision_score(labels_np, probs)
    metrics['accuracy'] = accuracy_score(labels_np, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(labels_np, preds)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels_np, preds).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return np.mean(losses), metrics


def print_metrics(metrics, prefix=''):
    """Print evaluation metrics"""
    print(f"\n{prefix} Metrics:")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    if 'ppv' in metrics and 'npv' in metrics:
        print(f"  PPV: {metrics['ppv']:.4f}")
        print(f"  NPV: {metrics['npv']:.4f}")
    print()


def save_checkpoint(state, filename):
    """Save checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    torch.cuda.set_device(args.gpu)
    
    print("="*70)
    print(f"MODE: {args.mode.upper()}")
    print("="*70)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    
    # Determine augmentation based on mode
    if args.mode == 'pretrain':
        dataloader_mode = 'pretrain'  # Will apply augmentation
    else:
        dataloader_mode = 'linear_eval'  # No augmentation
    
    dataloaders, datasets = get_dataloaders(
        skl_path=args.skl_path,
        csv_path=args.csv_path,
        biomarker=args.biomarker,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=dataloader_mode
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Get class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(datasets['train'])
        class_weights = class_weights.cuda()
    
    # Create model based on mode
    print("\nCreating model...")
    
    if args.mode == 'pretrain':
        print("Mode: Pretrain SupCon projection head")
        model = SupConEmbeddingModel(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            feat_dim=args.feat_dim,
            num_layers=args.num_layers
        ).cuda()
        
        criterion = SimpleSupConLoss(
            temperature=args.temperature
        ).cuda()
        
        training_mode = 'contrastive'
        
    elif args.mode == 'classifier':
        print("Mode: Train classifier directly on embeddings (NO pretrain)")
        print("   Input: Original TITAN embeddings (768D)")
        
        model = MLPClassifier(
            input_dim=args.input_dim,  # 768D
            hidden_dim=args.hidden_dim,
            num_classes=2
        ).cuda()
        
        # Setup criterion
        if args.focal_loss:
            criterion = FocalLoss(alpha=class_weights).cuda()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
        
        training_mode = 'classification'
        
    elif args.mode == 'finetune':
        print("Mode: Train classifier on SupCon-refined embeddings")
        
        if args.pretrained_supcon is None:
            raise ValueError("--pretrained_supcon is required for finetune mode")
        
        # Load pretrained SupCon model
        print(f"   Loading pretrained SupCon from: {args.pretrained_supcon}")
        supcon_model = SupConEmbeddingModel(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            feat_dim=args.feat_dim,
            num_layers=args.num_layers
        ).cuda()
        
        checkpoint = torch.load(args.pretrained_supcon, weights_only=False)
        if 'model_state_dict' in checkpoint:
            supcon_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            supcon_model.load_state_dict(checkpoint)
        supcon_model.eval()
        
        # Freeze SupCon model
        for param in supcon_model.parameters():
            param.requires_grad = False
        
        print("   Input: SupCon-refined embeddings (128D)")
        
        # Create classifier on refined embeddings
        model = MLPClassifier(
            input_dim=args.feat_dim,  # 128D (SupCon output)
            hidden_dim=args.hidden_dim,
            num_classes=2
        ).cuda()
        
        # Setup criterion
        if args.focal_loss:
            criterion = FocalLoss(alpha=class_weights).cuda()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
        
        training_mode = 'classification'
        
        # We'll need to wrap the forward pass to first pass through SupCon
        # This is handled in the training loop
    
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # Setup scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10
        )
    else:
        scheduler = None
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard'))
    
    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    best_val_metric = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        if training_mode == 'contrastive':
            train_loss = train_contrastive(
                train_loader, model, criterion, optimizer, epoch, args
            )
            print(f"Train Loss: {train_loss:.4f}")
            writer.add_scalar('train/loss', train_loss, epoch)
            
        else:  # classification
            train_loss, train_acc = train_classifier(
                train_loader, model, criterion, optimizer, epoch, args
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/accuracy', train_acc, epoch)
        
        # Validate
        val_loss, val_metrics = evaluate(
            val_loader, model, criterion, mode=training_mode, args=args
        )
        
        if training_mode == 'classification':
            print(f"Val Loss: {val_loss:.4f}")
            print_metrics(val_metrics, prefix='Validation')
            
            # Log to tensorboard
            writer.add_scalar('val/loss', val_loss, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            
            # Track best model
            val_metric = val_metrics['auroc']
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': args
                }, os.path.join(args.save_dir, 'best_model.pth'))
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Save final model
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    # Final test evaluation
    if training_mode == 'classification':
        print("\n" + "="*70)
        print("Final Test Evaluation")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics = evaluate(
            test_loader, model, criterion, mode='classification', args=args
        )
        print(f"Test Loss: {test_loss:.4f}")
        print_metrics(test_metrics, prefix='Test')
        
        # Save test results
        import json
        with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    writer.close()
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()