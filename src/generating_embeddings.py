import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import os
import pickle

HF_TOKEN = os.environ.get('HF_TOKEN', None)


# ============================================================================
# LoRA (same as training)
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
    """Add LoRA layers to the model"""
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
        """Encode slide using all patches"""
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


# ============================================================================
# Model Loading
# ============================================================================

def load_trained_model(
    checkpoint_path: str,
    model_name: str = 'MahmoodLab/TITAN',
    lora_r: int = 16,
    device: str = 'cuda'
):
    """
    Load the trained DPO model with LoRA weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        model_name: HuggingFace model name
        lora_r: LoRA rank (must match training)
        device: Device to load model on
    
    Returns:
        TITANSlideEncoder with loaded weights
    """
    print("="*70)
    print("Loading Trained DPO Model")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base model: {model_name}")
    print(f"LoRA rank: {lora_r}")
    print(f"Device: {device}")
    
    # Load base TITAN model
    print("\nLoading base TITAN model...")
    titan_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    titan_model = titan_model.to(device)
    
    # Freeze base model
    for param in titan_model.parameters():
        param.requires_grad = False
    
    # Add LoRA layers
    print("Adding LoRA layers...")
    lora_layers = add_lora(
        titan_model,
        rank=lora_r,
        alpha=lora_r,  # Typically alpha = rank
        device=device
    )
    print(f"   Added LoRA to {len(lora_layers)} layers")
    
    # Create encoder
    encoder = TITANSlideEncoder(titan_model, device=device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    encoder.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully!")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    if 'best_loss' in checkpoint:
        print(f"   Best training loss: {checkpoint['best_loss']:.4f}")
    
    # Set to eval mode
    encoder.eval()
    
    print("="*70)
    
    return encoder


# ============================================================================
# Inference Functions
# ============================================================================

def generate_slide_embedding(
    encoder: TITANSlideEncoder,
    patch_features: torch.Tensor,
    patch_coords: torch.Tensor,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Generate slide-level embedding from patch-level features.
    
    Args:
        encoder: Trained TITAN encoder
        patch_features: Patch features [num_patches, feature_dim]
        patch_coords: Patch coordinates [num_patches, 2]
        device: Device to use
    
    Returns:
        Slide embedding as numpy array [embedding_dim]
    """
    encoder.eval()
    
    with torch.no_grad():
        # Encode slide
        slide_emb = encoder(patch_features, patch_coords)
        
        # Convert to numpy
        slide_emb_np = slide_emb.cpu().numpy().squeeze()
    
    return slide_emb_np


def process_single_slide(
    encoder: TITANSlideEncoder,
    h5_path: Path,
    device: str = 'cuda'
) -> dict:
    """
    Process a single slide and generate embedding.
    
    Args:
        encoder: Trained encoder
        h5_path: Path to H5 file with patch features
        device: Device to use
    
    Returns:
        Dictionary with slide_id, embedding, and num_patches
    """
    slide_id = h5_path.stem
    
    # Load patch data
    with h5py.File(h5_path, 'r') as f:
        if 'embeddings' in f:
            patch_features = torch.from_numpy(f['embeddings'][:])
        elif 'features' in f:
            patch_features = torch.from_numpy(f['features'][:])
        else:
            raise KeyError(f"No embeddings/features found in {h5_path}")
        
        patch_coords = torch.from_numpy(f['coords'][:])
    
    num_patches = patch_features.shape[0]
    
    # Generate embedding
    slide_emb = generate_slide_embedding(
        encoder,
        patch_features,
        patch_coords,
        device
    )
    
    return {
        'slide_id': slide_id,
        'embedding': slide_emb,
        'num_patches': num_patches
    }


def batch_generate_embeddings(
    encoder: TITANSlideEncoder,
    patch_dir: str,
    output_path: str,
    slide_ids: list = None,
    device: str = 'cuda',
    save_format: str = 'pkl'
):
    """
    Generate embeddings for multiple slides.
    
    Args:
        encoder: Trained encoder
        patch_dir: Directory containing H5 files
        output_path: Path to save embeddings
        slide_ids: List of slide IDs to process (None = all)
        device: Device to use
        save_format: 'pkl', 'csv', or 'h5'
    """
    print("="*70)
    print("Batch Embedding Generation")
    print("="*70)
    
    patch_dir = Path(patch_dir)
    
    # Get list of H5 files
    if slide_ids is not None:
        h5_files = [patch_dir / f"{sid}.h5" for sid in slide_ids]
        h5_files = [f for f in h5_files if f.exists()]
    else:
        h5_files = list(patch_dir.glob("*.h5"))
    
    print(f"Found {len(h5_files)} slides to process")
    
    # Process all slides
    results = []
    
    for h5_path in tqdm(h5_files, desc="Generating embeddings"):
        try:
            result = process_single_slide(encoder, h5_path, device)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {h5_path.stem}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(results)} slides")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_format == 'pkl':
        # Save as pickle (preserves numpy arrays)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved embeddings to {output_path}")
    
    elif save_format == 'csv':
        # Save as CSV (embeddings as strings)
        df = pd.DataFrame({
            'slide_id': [r['slide_id'] for r in results],
            'embedding': [r['embedding'].tolist() for r in results],
            'num_patches': [r['num_patches'] for r in results]
        })
        df.to_csv(output_path, index=False)
        print(f"Saved embeddings to {output_path}")
    
    elif save_format == 'h5':
        # Save as HDF5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'slide_ids',
                data=np.array([r['slide_id'] for r in results], dtype='S')
            )
            f.create_dataset(
                'embeddings',
                data=np.stack([r['embedding'] for r in results])
            )
            f.create_dataset(
                'num_patches',
                data=np.array([r['num_patches'] for r in results])
            )
        print(f"Saved embeddings to {output_path}")
    
    else:
        raise ValueError(f"Unknown save format: {save_format}")
    
    # Print statistics
    print("\nEmbedding Statistics:")
    embeddings = np.stack([r['embedding'] for r in results])
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"   Std norm: {np.linalg.norm(embeddings, axis=1).std():.4f}")
    
    avg_patches = np.mean([r['num_patches'] for r in results])
    print(f"   Avg patches per slide: {avg_patches:.0f}")
    
    print("="*70)
    
    return results


# ============================================================================
# Comparison with Original Embeddings
# ============================================================================

def compare_embeddings(
    new_embeddings: list,
    original_csv: str,
    output_csv: str = None
):
    """
    Compare new DPO embeddings with original TITAN embeddings.
    
    Args:
        new_embeddings: List of dicts with slide_id and embedding
        original_csv: Path to CSV with original embeddings
        output_csv: Path to save comparison results
    """

    # Load original embeddings
    df_orig = pd.read_csv(original_csv, index_col=0)
    
    # Parse original embeddings
    def parse_embedding(emb_str):
        if isinstance(emb_str, str):
            try:
                import ast
                emb = ast.literal_eval(emb_str)
                return np.array(emb, dtype=np.float32)
            except:
                emb_str = emb_str.replace('\n', ' ').replace('  ', ' ')
                emb = np.fromstring(emb_str.strip('[]'), sep=' ', dtype=np.float32)
                return emb
        return np.array(emb_str, dtype=np.float32)
    
    df_orig['original_emb'] = df_orig['embeddings'].apply(parse_embedding)
    
    # Create new embeddings dict
    new_emb_dict = {r['slide_id']: r['embedding'] for r in new_embeddings}
    
    # Compare
    comparisons = []
    
    for idx, row in df_orig.iterrows():
        slide_id = row['slide_id']
        
        if slide_id not in new_emb_dict:
            continue
        
        orig_emb = row['original_emb']
        new_emb = new_emb_dict[slide_id]
        
        # Compute similarity
        similarity = np.dot(orig_emb, new_emb) / (
            np.linalg.norm(orig_emb) * np.linalg.norm(new_emb)
        )
        
        # Compute distance
        distance = np.linalg.norm(orig_emb - new_emb)
        
        comparisons.append({
            'slide_id': slide_id,
            'similarity': similarity,
            'distance': distance,
            'orig_norm': np.linalg.norm(orig_emb),
            'new_norm': np.linalg.norm(new_emb),
            'label': int(row['target'])
        })
    
    df_comp = pd.DataFrame(comparisons)
    
    # Print statistics
    print(f"\nCompared {len(df_comp)} slides")
    print(f"\nSimilarity Statistics:")
    print(f"   Mean: {df_comp['similarity'].mean():.4f}")
    print(f"   Std:  {df_comp['similarity'].std():.4f}")
    print(f"   Min:  {df_comp['similarity'].min():.4f}")
    print(f"   Max:  {df_comp['similarity'].max():.4f}")
    
    print(f"\nDistance Statistics:")
    print(f"   Mean: {df_comp['distance'].mean():.4f}")
    print(f"   Std:  {df_comp['distance'].std():.4f}")
    
    # By label
    print(f"\nSimilarity by Label:")
    for label in sorted(df_comp['label'].unique()):
        subset = df_comp[df_comp['label'] == label]
        print(f"   Label {label}: {subset['similarity'].mean():.4f} Â± {subset['similarity'].std():.4f}")
    
    # Save if requested
    if output_csv:
        df_comp.to_csv(output_csv, index=False)
        print(f"\n Saved comparison to {output_csv}")
    
    print("="*70)
    
    return df_comp


# ============================================================================
# Update CSV with New Embeddings
# ============================================================================

def update_csv_with_embeddings(
    original_csv: str,
    new_embeddings: list,
    output_csv: str,
    embedding_column: str = 'dpo_embeddings'
):
    """
    Update the original CSV with new embeddings.
    
    Args:
        original_csv: Original CSV file
        new_embeddings: List of dicts with slide_id and embedding
        output_csv: Output CSV path
        embedding_column: Name for new embedding column
    """
    print("\n" + "="*70)
    print("Updating CSV with New Embeddings")
    print("="*70)
    
    # Load original
    df = pd.read_csv(original_csv, index_col=0)
    
    # Create embedding dict
    emb_dict = {r['slide_id']: r['embedding'].tolist() for r in new_embeddings}
    
    # Add new column
    df[embedding_column] = df['slide_id'].map(emb_dict)
    
    # Count matches
    matched = df[embedding_column].notna().sum()
    print(f" Added embeddings for {matched}/{len(df)} slides")
    
    # Save
    df.to_csv(output_csv)
    print(f" Saved updated CSV to {output_csv}")
    print("="*70)
    
    return df


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate slide embeddings using trained DPO model'
    )
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--patch_dir', type=str, required=True,
                       help='Directory containing H5 patch files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for embeddings')
    
    # Optional
    parser.add_argument('--model_name', type=str, default='MahmoodLab/TITAN',
                       help='HuggingFace model name')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank (must match training)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save_format', type=str, default='pkl',
                       choices=['pkl', 'csv', 'h5'],
                       help='Output format')
    
    # Comparison
    parser.add_argument('--compare_with', type=str, default=None,
                       help='Original CSV to compare embeddings with')
    parser.add_argument('--comparison_output', type=str, default=None,
                       help='Output path for comparison results')
    
    # Update CSV
    parser.add_argument('--update_csv', type=str, default=None,
                       help='CSV file to update with new embeddings')
    parser.add_argument('--updated_csv_output', type=str, default=None,
                       help='Output path for updated CSV')
    
    args = parser.parse_args()
    
    # Load model
    encoder = load_trained_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        lora_r=args.lora_r,
        device=args.device
    )
    
    # Generate embeddings
    results = batch_generate_embeddings(
        encoder=encoder,
        patch_dir=args.patch_dir,
        output_path=args.output,
        device=args.device,
        save_format=args.save_format
    )
    
    # Compare if requested
    if args.compare_with:
        compare_embeddings(
            new_embeddings=results,
            original_csv=args.compare_with,
            output_csv=args.comparison_output
        )
    
    # Update CSV if requested
    if args.update_csv:
        output_path = args.updated_csv_output or args.update_csv.replace('.csv', '_updated.csv')
        update_csv_with_embeddings(
            original_csv=args.update_csv,
            new_embeddings=results,
            output_csv=output_path
        )
    
    print("\n All done!")


if __name__ == '__main__':
    main()