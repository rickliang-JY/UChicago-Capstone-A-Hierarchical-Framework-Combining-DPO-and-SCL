"""
Simple slide-level random splitting (without patient-level consideration)
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn


class SlideLevelDataset(Dataset):
    """Dataset for slide-level embeddings"""
    
    def __init__(self, embeddings_dict, clinical_df, biomarker, 
                 slide_ids, augment=True, aug_strength='medium'):
        """
        Args:
            embeddings_dict: Dictionary {slide_id: embedding_array}
            clinical_df: DataFrame with columns [slide_id, submitter_id, biomarker]
            biomarker: Which biomarker to predict
            slide_ids: List of slide IDs to include in this split
            augment: Whether to apply embedding augmentation
            aug_strength: 'weak', 'medium', or 'strong'
        """
        self.embeddings_dict = embeddings_dict
        self.biomarker = biomarker
        self.augment = augment
        self.aug_strength = aug_strength
        
        # Filter to only include specified slides
        clinical_df = clinical_df[clinical_df['slide_id'].isin(slide_ids)]
        clinical_df = clinical_df[clinical_df[biomarker].notna()].reset_index(drop=True)
        
        self.clinical_df = clinical_df
        self.slide_ids = clinical_df['slide_id'].tolist()
        
        # Handle labels - convert strings to binary if needed
        raw_labels = clinical_df[biomarker].values
        
        if isinstance(raw_labels[0], str):
            print(f"  Converting string labels to binary...")
            positive_values = ['positive', 'Positive', 'pos', 'Pos', '1', 'yes', 'Yes', 'TRUE', 'True']
            negative_values = ['negative', 'Negative', 'neg', 'Neg', '0', 'no', 'No', 'FALSE', 'False']
            
            labels = []
            for val in raw_labels:
                if val in positive_values:
                    labels.append(1)
                elif val in negative_values:
                    labels.append(0)
                else:
                    raise ValueError(f"Unknown label value: '{val}'")
            
            self.labels = np.array(labels, dtype=int)
        else:
            self.labels = raw_labels.astype(int)
        
        self.submitter_ids = clinical_df['submitter_id'].values
        
        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        label_names = {0: 'negative', 1: 'positive'}
        distribution = {label_names[u]: c for u, c in zip(unique, counts)}
        print(f"{biomarker} distribution: {distribution}")
        print(f"Total slides: {len(self.slide_ids)}")
    
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        
        # Get slide-level embedding
        embedding = self.embeddings_dict[slide_id]
        
        # Convert to tensor
        embedding = torch.from_numpy(embedding).float()
        label = torch.tensor(label, dtype=torch.long)
        
        if self.augment:
            # Return two augmented views for contrastive learning
            view1 = self._augment_embedding(embedding)
            view2 = self._augment_embedding(embedding)
            return [view1, view2], label
        else:
            return embedding, label
    
    def _augment_embedding(self, embedding):
        """Augmentation strategies for slide-level embeddings"""
        augmented = embedding.clone()
        
        # Set augmentation strength
        if self.aug_strength == 'weak':
            noise_std = 0.01
            dropout_rate = 0.05
            scale_range = (0.98, 1.02)
        elif self.aug_strength == 'medium':
            noise_std = 0.05
            dropout_rate = 0.15
            scale_range = (0.90, 1.10)
        else:  # strong
            noise_std = 0.10
            dropout_rate = 0.20
            scale_range = (0.85, 1.15)
        
        # 1. Gaussian noise
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
        
        # 2. Dropout
        dropout_mask = torch.bernoulli(torch.ones_like(augmented) * (1 - dropout_rate))
        augmented = augmented * dropout_mask
        
        # 3. Random scaling
        scale = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
        augmented = augmented * scale
        
        # 4. Re-normalize
        augmented = F.normalize(augmented, dim=0, p=2)
        
        return augmented


class SimpleSupConLoss(nn.Module):
    """Standard Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SimpleSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [bsz, n_views, feat_dim]
            labels: [bsz]
            mask: [bsz, bsz]
        Returns:
            loss: scalar
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, feat_dim]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


def load_embeddings_from_skl(skl_path):
    """Load embeddings from pickle/SKL file"""
    print(f"Loading embeddings from {skl_path}")
    
    with open(skl_path, 'rb') as file:
        data = pickle.load(file)
    
    if isinstance(data, dict):
        embeddings_dict = data
    elif isinstance(data, tuple) and len(data) == 2:
        embeddings, slide_ids = data
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(-1, 1)
            embeddings_dict = {
                slide_id: embeddings[i]
                for i, slide_id in enumerate(slide_ids)
            }
        else:
            embeddings_dict = dict(zip(slide_ids, embeddings))
    elif isinstance(data, pd.DataFrame):
        if 'slide_id' not in data.columns:
            raise ValueError(f"DataFrame must have 'slide_id' column")
        
        other_cols = [col for col in data.columns if col != 'slide_id']
        
        if len(other_cols) == 1:
            embedding_col = other_cols[0]
            embeddings_dict = {}
            for idx, row in data.iterrows():
                slide_id = row['slide_id']
                embedding = row[embedding_col]
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                else:
                    embedding = embedding.astype(np.float32)
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                embeddings_dict[slide_id] = embedding
        else:
            embeddings_dict = {}
            for idx, row in data.iterrows():
                slide_id = row['slide_id']
                embedding = row[other_cols].values.astype(np.float32)
                embeddings_dict[slide_id] = embedding
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    print(f"Loaded {len(embeddings_dict)} slide embeddings")
    sample_embedding = next(iter(embeddings_dict.values()))
    print(f"Embedding dimension: {sample_embedding.shape}")
    
    return embeddings_dict


def load_clinical_data(csv_path, biomarker):
    """Load clinical data"""
    print(f"Loading clinical data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    required_cols = ['slide_id', 'submitter_id', biomarker]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} samples")
    return df


def create_simple_random_splits(clinical_df, embeddings_dict, biomarker, 
                                test_size=0.25, val_size=0.25, random_state=42):
    """
    Simple random split 
    1. temp_df, test = train_test_split(df, test_size=0.25, random_state=42)
    2. train, val = train_test_split(temp_df, test_size=1/3, random_state=42)
    """
    print("\n" + "="*70)
    print("SIMPLE RANDOM SLIDE-LEVEL SPLITTING")
    print("="*70)
    print(f"Split ratios: Train ~50%, Val ~25%, Test 25%")
    print("="*70)
    
    # Keep only samples with valid embeddings and labels
    valid_df = clinical_df[
        (clinical_df['slide_id'].isin(embeddings_dict.keys())) &
        (clinical_df[biomarker].notna())
    ].copy()
    
    # Get labels for stratification
    labels = valid_df[biomarker].values
    
    # Convert labels to binary if needed
    if isinstance(labels[0], str):
        positive_values = ['positive', 'Positive', 'pos', 'Pos', '1', 'yes', 'Yes', 'TRUE', 'True']
        labels = np.array([1 if x in positive_values else 0 for x in labels])
    else:
        labels = labels.astype(int)
    
    # ç¬¬ä¸€æ­¥: temp_df, test = train_test_split(df, test_size=0.25, random_state=42)
    temp_df, test_df = train_test_split(
        valid_df, 
        test_size=test_size,  # 0.25
        random_state=random_state,
        stratify=labels
    )
    
    # ç¬¬äºŒæ­¥: train, val = train_test_split(temp_df, test_size=1/3, random_state=42)
    # 1/3 of temp_df = 1/3 * 0.75 = 0.25 of total
    temp_labels = temp_df[biomarker].values
    if isinstance(temp_labels[0], str):
        positive_values = ['positive', 'Positive', 'pos', 'Pos', '1', 'yes', 'Yes', 'TRUE', 'True']
        temp_labels = np.array([1 if x in positive_values else 0 for x in temp_labels])
    else:
        temp_labels = temp_labels.astype(int)
    
    train_df, val_df = train_test_split(
        temp_df,
        test_size=1/3,  # å›ºå®š 1/3
        random_state=random_state,
        stratify=temp_labels
    )
    
    # Extract slide IDs
    splits = {
        'train': train_df['slide_id'].values,
        'val': val_df['slide_id'].values,
        'test': test_df['slide_id'].values
    }
    
    # Print split statistics
    for split_name, slide_ids in splits.items():
        split_df = valid_df[valid_df['slide_id'].isin(slide_ids)]
        n_slides = len(split_df)
        n_patients = split_df['submitter_id'].nunique()
        
        split_labels = split_df[biomarker].values
        if isinstance(split_labels[0], str):
            positive_values = ['positive', 'Positive', 'pos', 'Pos', '1', 'yes', 'Yes', 'TRUE', 'True']
            split_labels = np.array([1 if x in positive_values else 0 for x in split_labels])
        else:
            split_labels = split_labels.astype(int)
        
        unique, counts = np.unique(split_labels, return_counts=True)
        label_dist = {
            'positive' if l == 1 else 'negative': c 
            for l, c in zip(unique, counts)
        }
        
        print(f"\n{split_name.upper()}:")
        print(f"  Slides: {n_slides} ({n_slides/len(valid_df)*100:.1f}%)")
        print(f"  Patients: {n_patients}")
        print(f"  Distribution: {label_dist}")
    
    return splits


def normalize_embeddings(embeddings_dict):
    """Normalize embeddings (L2 normalization)"""
    print("\nNormalizing embeddings...")
    normalized_embeddings = {}
    for slide_id, embedding in embeddings_dict.items():
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized_embeddings[slide_id] = embedding / norm
        else:
            print(f"WARNING: Zero norm for slide {slide_id}")
            normalized_embeddings[slide_id] = embedding
    
    print(f"âœ“ Normalized {len(normalized_embeddings)} embeddings")
    return normalized_embeddings


def get_dataloaders(skl_path, csv_path, biomarker, batch_size=32, 
                   num_workers=4, test_size=0.25, val_size=0.25,
                   random_state=42, mode='pretrain', aug_strength='strong'):
    """
    Create train/val/test dataloaders with simple random splitting
    - temp_df, test = train_test_split(df, test_size=0.25, random_state=42)
    - train, val = train_test_split(temp_df, test_size=1/3, random_state=42)
    """
    # Load embeddings
    embeddings_dict = load_embeddings_from_skl(skl_path)
    
    # Normalize embeddings
    embeddings_dict = normalize_embeddings(embeddings_dict)
    
    # Load clinical data
    clinical_df = load_clinical_data(csv_path, biomarker)
    
    # Create simple random splits
    splits = create_simple_random_splits(
        clinical_df, embeddings_dict, biomarker, 
        test_size, val_size, random_state
    )
    
    # Create datasets
    datasets = {}
    for split_name, slide_ids in splits.items():
        # Only augment training set in pretrain mode
        augment = (split_name == 'train' and mode == 'pretrain')
        
        datasets[split_name] = SlideLevelDataset(
            embeddings_dict=embeddings_dict,
            clinical_df=clinical_df,
            biomarker=biomarker,
            slide_ids=slide_ids,
            augment=augment,
            aug_strength=aug_strength
        )
    
    # Create dataloaders
    dataloaders = {}
    for split_name, dataset in datasets.items():
        shuffle = (split_name == 'train')
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == 'train')
        )
    
    return dataloaders, datasets


def get_class_weights(dataset):
    """Compute class weights for imbalanced data"""
    labels = dataset.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Inverse frequency weighting
    weights = len(labels) / (len(unique_labels) * counts)
    weight_dict = dict(zip(unique_labels, weights))
    
    print(f"Class weights: {weight_dict}")
    
    # Convert to tensor
    class_weights = torch.FloatTensor([weight_dict[i] for i in sorted(unique_labels)])
    
    return class_weights