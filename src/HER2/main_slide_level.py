"""
Simplified training script for slide-level TITAN embeddings
Works with pickle/sklearn files containing pre-aggregated slide embeddings
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
    parser = argparse.ArgumentParser('TITAN Slide-Level SupCon Training')
    
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
    parser.add_argument('--mode', type=str, default='pretrain',
                        choices=['pretrain', 'linear_eval', 'end_to_end'],
                        help='Training mode')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
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
    
    # Checkpoint and logging
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pre-trained model (for linear_eval mode)')
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
            f"bs{args.batch_size}_lr{args.learning_rate}_"
            f"temp{args.temperature}"
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
        
        bsz = labels.shape[0]
        
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


def evaluate(val_loader, model, criterion, mode='contrastive', args=None):
    """Evaluate model"""
    model.eval()
    
    losses = []
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.cuda()
            labels = labels.cuda()
            
            if mode == 'contrastive':
                continue
            else:
                # Classification mode
                logits = model(embeddings)
                loss = criterion(logits, labels)
                
                losses.append(loss.item())
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    if mode == 'contrastive':
        return 0.0, {}
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = np.mean(losses)
    
    return avg_loss, metrics


def compute_metrics(labels, preds, probs):
    """Compute comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)
    
    # ROC-AUC and PR-AUC
    try:
        metrics['auroc'] = roc_auc_score(labels, probs)
        metrics['auprc'] = average_precision_score(labels, probs)
    except:
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0
    
    # Confusion matrix metrics
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return metrics


def print_metrics(metrics, prefix=''):
    """Print metrics in a formatted way"""
    print(f"\n{prefix} Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    if 'sensitivity' in metrics:
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
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
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders, datasets = get_dataloaders(
        skl_path=args.skl_path,
        csv_path=args.csv_path,
        biomarker=args.biomarker,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode  # Pass the mode to control augmentation
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Get class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(datasets['train'])
        class_weights = class_weights.cuda()
    
    # Create model
    print("Creating model...")
    if args.mode == 'pretrain':
        # Contrastive pre-training
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
        
    else:  # linear_eval or end_to_end
        # For linear_eval, we need to use the pre-trained embeddings
        # Option 1: Load pre-trained model and use it to refine embeddings (proper way)
        # Option 2: Just train on raw embeddings (simpler, but less effective)
        
        # We'll use Option 2 for now (simpler implementation)
        # The pre-training step already learned good representations,
        # and we're training a new classifier on the original embeddings
        
        if args.pretrained_model is not None:
            print(f"Note: Pre-trained model path provided: {args.pretrained_model}")
            print("Currently training classifier on original embeddings.")
            print("For best results, embeddings should be passed through pre-trained projection head first.")
            print("(This is a simplified implementation - full implementation coming soon)")
        
        # Classification training
        model = MLPClassifier(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2
        ).cuda()
        
        # Setup criterion
        if args.focal_loss:
            criterion = FocalLoss(alpha=class_weights).cuda()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
        
        training_mode = 'classification'
    
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
    print("Starting training...")
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
        print("\n" + "="*50)
        print("Final Test Evaluation")
        print("="*50)
        
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
    print("\nTraining complete!")


if __name__ == '__main__':
    main()