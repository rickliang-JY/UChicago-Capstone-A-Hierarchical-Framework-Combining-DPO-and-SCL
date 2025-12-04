"""
Model architectures for TITAN slide embedding refinement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingProjector(nn.Module):
    """
    Projection head for embedding refinement
    Maps TITAN embeddings to a lower-dimensional space for contrastive learning
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128, 
                 num_layers=2, use_bn=True):
        """
        Args:
            input_dim: Dimension of TITAN embeddings (768 for TITAN)
            hidden_dim: Hidden dimension
            output_dim: Output dimension for contrastive space
            num_layers: Number of layers in projection head (2 or 3)
            use_bn: Whether to use batch normalization
        """
        super(EmbeddingProjector, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if num_layers == 2:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            ]
        elif num_layers == 3:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            ]
        else:
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] embeddings
        Returns:
            [batch_size, output_dim] projected embeddings
        """
        # Project
        out = self.projector(x)
        # L2 normalize for contrastive learning
        out = F.normalize(out, dim=1)
        return out


class SupConEmbeddingModel(nn.Module):
    """
    Full model for supervised contrastive learning on embeddings
    
    Architecture:
        Input embeddings -> Projection head -> Normalized features
    """
    def __init__(self, input_dim=768, hidden_dim=512, feat_dim=128, 
                 num_layers=2, use_bn=True):
        super(SupConEmbeddingModel, self).__init__()
        
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        
        # Identity "encoder" (embeddings are pre-extracted)
        # This is just for compatibility with the original SupCon structure
        self.encoder = nn.Identity()
        
        # Projection head
        # self.head = EmbeddingProjector(
        #     input_dim=input_dim,
        #     hidden_dim=hidden_dim,
        #     output_dim=feat_dim,
        #     num_layers=num_layers,
        #     use_bn=use_bn
        # )
        self.head = EmbeddingRefinementModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=feat_dim,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] pre-extracted embeddings
        Returns:
            [batch_size, feat_dim] normalized projected features
        """
        # "Encode" (identity operation since embeddings are pre-extracted)
        feat = self.encoder(x)
        
        # Project to contrastive space
        feat = self.head(feat)
        
        return feat


class LinearClassifier(nn.Module):
    """
    Linear classifier for evaluation on frozen embeddings
    """
    def __init__(self, input_dim=768, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    """
    MLP classifier for evaluation (more capacity than linear)
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, 
                 dropout=0.5):
        super(MLPClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class SupConWithClassifier(nn.Module):
    """
    Complete model for joint contrastive + classification training
    (Alternative to two-stage approach)
    """
    def __init__(self, input_dim=768, hidden_dim=512, feat_dim=128,
                 num_classes=2, num_layers=2):
        super(SupConWithClassifier, self).__init__()
        
        # Shared encoder (identity for pre-extracted embeddings)
        self.encoder = nn.Identity()
        
        # Projection head for contrastive learning
        self.projection_head = EmbeddingProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=feat_dim,
            num_layers=num_layers
        )
        
        # Classification head
        self.classification_head = nn.Linear(input_dim, num_classes)
    
    def forward(self, x, return_projection=False):
        """
        Args:
            x: [batch_size, input_dim] embeddings
            return_projection: If True, return both logits and projected features
        
        Returns:
            logits: [batch_size, num_classes]
            projected (optional): [batch_size, feat_dim]
        """
        # "Encode"
        feat = self.encoder(x)
        
        # Classification
        logits = self.classification_head(feat)
        
        if return_projection:
            # Projection for contrastive loss
            projected = self.projection_head(feat)
            return logits, projected
        
        return logits


class EmbeddingRefinementModel(nn.Module):
    """
    Model that refines embeddings through a learned transformation
    Can be used to improve pre-extracted TITAN embeddings
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768,
                 num_layers=2, residual=True):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (usually same as input)
            num_layers: Number of transformation layers
            residual: Whether to use residual connection
        """
        super(EmbeddingRefinementModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual and (input_dim == output_dim)
        
        if num_layers == 1:
            self.transform = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif num_layers == 2:
            self.transform = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif num_layers == 3:
            self.transform = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            raise ValueError(f"num_layers must be 1, 2, or 3, got {num_layers}")
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            [batch_size, output_dim] refined embeddings
        """
        out = self.transform(x)
        
        if self.residual:
            print("Use Residual")
            out = out + x
        
        # Normalize to preserve embedding space properties
        out = F.normalize(out, dim=1)
        
        return out


class AdapterModule(nn.Module):
    """
    Lightweight adapter module for embedding refinement
    Inspired by adapter layers in NLP fine-tuning
    """
    def __init__(self, input_dim=768, bottleneck_dim=64):
        """
        Args:
            input_dim: Embedding dimension
            bottleneck_dim: Bottleneck dimension (typically much smaller)
        """
        super(AdapterModule, self).__init__()
        
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            [batch_size, input_dim] adapted embeddings
        """
        # Adapter transformation
        h = self.down_project(x)
        h = self.activation(h)
        h = self.up_project(h)
        
        # Residual connection
        out = x + h
        
        # Layer norm
        out = self.layer_norm(out)
        
        return out


def load_pretrained_encoder(model_path, model):
    """
    Load pre-trained SupCon model for linear evaluation
    
    Args:
        model_path: Path to saved model checkpoint
        model: Model to load weights into
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load encoder and projection head
    model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded pre-trained model from {model_path}")
    
    return model


def freeze_encoder(model):
    """Freeze encoder weights (only train projection head)"""
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen")


def unfreeze_all(model):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True
    print("All parameters unfrozen")