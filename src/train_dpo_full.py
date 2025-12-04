import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import ast
import os
from datetime import datetime
HF_TOKEN = os.environ.get('HF_TOKEN', None)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# LoRA
# ============================================================================

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, original_layer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        if original_layer is not None:
            self.weight = original_layer.weight
            self.bias = original_layer.bias
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        result = result + lora_out * self.scaling
        return result


def add_lora(model, rank=8, alpha=16, device='cuda'):
    lora_layers = {}
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_replace.append((name, module))
    
    for name, module in modules_to_replace:
        *parent_path, attr_name = name.split('.')
        
        if parent_path:
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
        else:
            parent = model
        
        lora_layer = LoRALinear(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
            original_layer=module
        )
        lora_layer = lora_layer.to(device)
        setattr(parent, attr_name, lora_layer)
        lora_layers[name] = lora_layer
    
    print(f"   Added LoRA to {len(lora_layers)} layers")
    return lora_layers


# ============================================================================
# TITAN Encoder
# ============================================================================

class TITANSlideEncoder(nn.Module):
    def __init__(self, titan_model, patch_size_lv0=256, device='cuda'):
        super().__init__()
        self.titan_model = titan_model
        self.patch_size_lv0 = patch_size_lv0
        self.device = device
    
    def encode_slide(self, patch_features, patch_coords):
        """Encode slide - use all patches, no restriction."""
        patch_features = patch_features.to(self.device)
        patch_coords = patch_coords.to(self.device)
        
        slide_embedding = self.titan_model.encode_slide_from_patch_features(
            patch_features=patch_features,
            patch_coords=patch_coords,
            patch_size_lv0=self.patch_size_lv0
        )
        
        return slide_embedding
    
    def forward(self, patches_or_features, patch_coords):
        return self.encode_slide(patches_or_features, patch_coords)


def load_titan_model(model_name='MahmoodLab/TITAN', device='cuda', freeze=True):
    print(f"Loading TITAN model from: {model_name}")
    
    titan_model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        token=HF_TOKEN
    )
    titan_model = titan_model.to(device)
    
    if freeze:
        print("Freezing TITAN backbone")
        for param in titan_model.parameters():
            param.requires_grad = False
        titan_model.eval()
    
    encoder = TITANSlideEncoder(titan_model, device=device)
    print("Model loaded successfully\n")
    
    return encoder




class FullPatchDPODataset(Dataset):
    def __init__(self, csv_path: str, patch_dir: str):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.patch_dir = Path(patch_dir)
        
        self.df['embeddings_array'] = self.df['embeddings'].apply(self._parse_embedding)
        self.df['cluster_center_array'] = self.df['cluster_center_embedding'].apply(self._parse_embedding)
        
        print(f"Dataset loaded: {len(self.df)} slides")
        
        self._analyze_patch_counts()
    
    def _parse_embedding(self, emb_str):
        if isinstance(emb_str, str):
            try:
                emb = ast.literal_eval(emb_str)
                return np.array(emb, dtype=np.float32)
            except:
                emb_str = emb_str.replace('\n', ' ').replace('  ', ' ')
                emb = np.fromstring(emb_str.strip('[]'), sep=' ', dtype=np.float32)
                return emb
        return np.array(emb_str, dtype=np.float32)
    
    def _analyze_patch_counts(self):
        print("\nAnalyzing patch counts...")
        patch_counts = []
        
        for idx, row in self.df.iterrows():
            h5_path = self.patch_dir / f"{row['slide_id']}.h5"
            if h5_path.exists():
                with h5py.File(h5_path, 'r') as f:
                    if 'embeddings' in f:
                        count = f['embeddings'].shape[0]
                    elif 'features' in f:
                        count = f['features'].shape[0]
                    else:
                        count = 0
                    patch_counts.append(count)
        
        if patch_counts:
            print(f"   Total slides with patches: {len(patch_counts)}")
            print(f"   Patch count statistics:")
            print(f"      Mean:   {np.mean(patch_counts):.0f}")
            print(f"      Median: {np.median(patch_counts):.0f}")
            print(f"      Min:    {np.min(patch_counts):.0f}")
            print(f"      Max:    {np.max(patch_counts):.0f}")
            print(f"      Std:    {np.std(patch_counts):.0f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        h5_path = self.patch_dir / f"{slide_id}.h5"
        
        with h5py.File(h5_path, 'r') as f:
            if 'embeddings' in f:
                data = torch.from_numpy(f['embeddings'][:])  # all patches!
            elif 'features' in f:
                data = torch.from_numpy(f['features'][:])
            else:
                raise KeyError(f"No embeddings found in {h5_path}")
            coords = torch.from_numpy(f['coords'][:])
        
        return {
            'slide_id': slide_id,
            'data': data, 
            'coords': coords,
            'num_patches': data.shape[0],  # number of patches
            'preferred_emb': torch.from_numpy(row['cluster_center_array']),
            'rejected_emb': torch.from_numpy(row['embeddings_array']),
            'label': int(row['target'])
        }


def collate_fn_full_patches(batch):
    """Collate function for full patches - no truncation"""
    return {
        'slide_id': [item['slide_id'] for item in batch],
        'data': [item['data'] for item in batch],
        'coords': [item['coords'] for item in batch],
        'num_patches': [item['num_patches'] for item in batch],
        'preferred_emb': torch.stack([item['preferred_emb'] for item in batch]),
        'rejected_emb': torch.stack([item['rejected_emb'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch])
    }


# ============================================================================
# DPO Loss
# ============================================================================

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.5
):
 
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    return losses, chosen_rewards, rejected_rewards




class FullPatchDPOTrainer:
    """Train with full patches; supports gradient accumulation."""
    
    def __init__(
        self,
        titan_encoder,
        ref_titan_encoder,
        beta=0.5,
        learning_rate=1e-5,
        device='cuda',
        use_lora=True,
        lora_r=16,
        ref_on_cpu=True,
        gradient_accumulation_steps=1,  # Added: gradient accumulation
        max_grad_norm=1.0
    ):
        self.titan_encoder = titan_encoder
        self.ref_titan_encoder = ref_titan_encoder
        self.beta = beta
        self.device = device
        self.ref_on_cpu = ref_on_cpu
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.train_losses = []
        
        # Reference model setup
        if ref_on_cpu:
            self.ref_titan_encoder = self.ref_titan_encoder.cpu()
            self.ref_device = 'cpu'
        else:
            self.ref_device = device
        
        self.ref_titan_encoder.eval()
        for param in self.ref_titan_encoder.parameters():
            param.requires_grad = False
        
        # Apply LoRA
        if use_lora:
            print(" Applying LoRA adapters...")
            self.lora_layers = add_lora(
                titan_encoder.titan_model, 
                rank=lora_r, 
                alpha=lora_r,
                device=device
            )
        
        # Optimizer
        trainable_params = []
        if use_lora and hasattr(self, 'lora_layers'):
            for name, lora in self.lora_layers.items():
                trainable_params.extend([lora.lora_A, lora.lora_B])
        else:
            trainable_params = list(self.titan_encoder.parameters())
        
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        print(f"\nFull Patch DPO Trainer initialized")
        print(f"   Beta: {beta}")
        print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"   Max grad norm: {max_grad_norm}")
        print(f"   Using ALL patches (no truncation)")
    
    def compute_dpo_loss(self, batch):
        """Compute DPO loss - using full patches."""
        
        policy_chosen_logps_list = []
        policy_rejected_logps_list = []
        ref_chosen_logps_list = []
        ref_rejected_logps_list = []
        
        for i in range(len(batch['data'])):
            data = batch['data'][i]
            coords = batch['coords'][i]
            num_patches = batch['num_patches'][i]
            
            # Move to device (use all patches)
            data = data.to(self.device)
            coords = coords.to(self.device)
            
            # Policy model forward - full patches
            policy_emb = self.titan_encoder(data, coords)
            
            # Chosen (preferred)
            preferred_emb = batch['preferred_emb'][i].to(self.device)
            chosen_sim = F.cosine_similarity(policy_emb, preferred_emb.unsqueeze(0), dim=-1)
            policy_chosen_logps = torch.log(torch.sigmoid(chosen_sim * 10))
            
            # Rejected
            rejected_emb = batch['rejected_emb'][i].to(self.device)
            rejected_sim = F.cosine_similarity(policy_emb, rejected_emb.unsqueeze(0), dim=-1)
            policy_rejected_logps = torch.log(torch.sigmoid(rejected_sim * 10))
            
            policy_chosen_logps_list.append(policy_chosen_logps)
            policy_rejected_logps_list.append(policy_rejected_logps)
            
            # Reference model forward - full patches
            with torch.no_grad():
                if self.ref_on_cpu:
                    data_cpu = data.cpu()
                    coords_cpu = coords.cpu()
                    ref_emb = self.ref_titan_encoder(data_cpu, coords_cpu)
                    ref_emb = ref_emb.to(self.device)
                    del data_cpu, coords_cpu
                else:
                    ref_emb = self.ref_titan_encoder(data, coords)
                
                ref_chosen_sim = F.cosine_similarity(ref_emb, preferred_emb.unsqueeze(0), dim=-1)
                ref_chosen_logps = torch.log(torch.sigmoid(ref_chosen_sim * 10))
                
                ref_rejected_sim = F.cosine_similarity(ref_emb, rejected_emb.unsqueeze(0), dim=-1)
                ref_rejected_logps = torch.log(torch.sigmoid(ref_rejected_sim * 10))
                
                ref_chosen_logps_list.append(ref_chosen_logps)
                ref_rejected_logps_list.append(ref_rejected_logps)
            
            del data, coords
        
        # Stack
        policy_chosen_logps = torch.stack(policy_chosen_logps_list)
        policy_rejected_logps = torch.stack(policy_rejected_logps_list)
        ref_chosen_logps = torch.stack(ref_chosen_logps_list)
        ref_rejected_logps = torch.stack(ref_rejected_logps_list)
        
        # DPO loss
        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta
        )
        
        loss = losses.mean()
        
        # Metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        metrics = {
            'loss': loss.item(),
            'rewards/chosen': chosen_rewards.mean().item(),
            'rewards/rejected': rejected_rewards.mean().item(),
            'rewards/accuracies': reward_accuracies.mean().item(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean().item(),
            'avg_num_patches': np.mean(batch['num_patches'])
        }
        
        return loss, metrics
    
    def train_epoch(self, dataloader, epoch):
        self.titan_encoder.train()
        self.ref_titan_encoder.eval()
        
        epoch_losses = []
        epoch_metrics = {
            'rewards/chosen': [],
            'rewards/rejected': [],
            'rewards/accuracies': [],
            'rewards/margins': [],
            'avg_num_patches': []
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Compute loss
                loss, metrics = self.compute_dpo_loss(batch)
                
                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Update parameters
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.titan_encoder.parameters(), 
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Log
                epoch_losses.append(metrics['loss'])
                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])
                
                # Update progress
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['rewards/accuracies']:.2%}",
                    'margin': f"{metrics['rewards/margins']:.3f}",
                    'patches': f"{metrics['avg_num_patches']:.0f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Handle leftover accumulation if dataloader size not divisible by accumulation steps
        if len(dataloader) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.titan_encoder.parameters(), 
                self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        self.train_losses.append(avg_loss)
        
        summary = {
            'loss': avg_loss,
            **{k: np.mean(v) for k, v in epoch_metrics.items()}
        }
        
        return avg_loss, summary
    
    def save_checkpoint(self, save_path, epoch, additional_info=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.titan_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
        }
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        print(f"Saved: {save_path}")


# ============================================================================
# Training Function
# ============================================================================

def train(
    csv_path,
    patch_dir,
    model_name='MahmoodLab/TITAN',
    num_epochs=20,
    batch_size=1,  #  the number of patches per slide varies
    learning_rate=1e-5,
    beta=0.5,
    use_lora=True,
    lora_r=16,
    device='cuda',
    save_dir='./checkpoints_full_patch',
    ref_on_cpu=True,
    gradient_accumulation_steps=1,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TITAN DPO Training - Full Patch Mode")
    print("=" * 70)
    print("Using ALL patches (no max_patches truncation)")
    print("   This may require more memory")
    print("   Consider using gradient accumulation if OOM")
    print("=" * 70)
    
    # Load models
    titan_encoder = load_titan_model(model_name, device, freeze=use_lora)
    
    ref_titan_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
    for param in ref_titan_model.parameters():
        param.requires_grad = False
    ref_device = 'cpu' if ref_on_cpu else device
    ref_titan_encoder = TITANSlideEncoder(ref_titan_model, device=ref_device)
    ref_titan_encoder.eval()
    
    # Data
    dataset = FullPatchDPODataset(csv_path, patch_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=collate_fn_full_patches
    )
    
    # Trainer
    trainer = FullPatchDPOTrainer(
        titan_encoder=titan_encoder,
        ref_titan_encoder=ref_titan_encoder,
        beta=beta,
        learning_rate=learning_rate,
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        ref_on_cpu=ref_on_cpu,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Training loop
    print("\nStarting training...\n")
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        avg_loss, metrics = trainer.train_epoch(dataloader, epoch)
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Reward Accuracy: {metrics['rewards/accuracies']:.2%}")
        print(f"   Reward Margin: {metrics['rewards/margins']:.4f}")
        print(f"   Avg Patches per Slide: {metrics['avg_num_patches']:.0f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(save_dir) / f'epoch{num_epochs}_lr{learning_rate}_beta{beta}_best_model.pt'
            trainer.save_checkpoint(save_path, epoch, {'best_loss': best_loss, 'metrics': metrics})
        
        if epoch % 5 == 0:
            save_path = Path(save_dir) / f'epoch{num_epochs}_lr{learning_rate}_beta{beta}_checkpoint_epoch_{epoch}_{timestamp}.pt'
            trainer.save_checkpoint(save_path, epoch)
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    return titan_encoder, trainer


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TITAN DPO - Full Patch Mode')
    
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--patch_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='MahmoodLab/TITAN')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--ref_on_cpu', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (increase if OOM)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_full_patch')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train(
        csv_path=args.csv_path,
        patch_dir=args.patch_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        device=args.device,
        save_dir=args.save_dir,
        ref_on_cpu=args.ref_on_cpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )