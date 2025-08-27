import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import json
import os
from pathlib import Path
import time
import psutil

# GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available.")


class TextEncoder(nn.Module):
    """Text encoder using DistilBERT"""
    
    def __init__(self, output_dim=256, dropout=0.2):
        super().__init__()
        
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze layers 0-3, unfreeze layers 4-5
        for i, layer in enumerate(self.distilbert.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # CLS token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return self.projection(cls_output)


class VideoEncoder(nn.Module):
    """Video encoder using ResNet50"""
    
    def __init__(self, output_dim=256, dropout=0.2):
        super().__init__()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all except layer4 and fc
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        
        self.resnet.fc = nn.Identity()
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
        
    def forward(self, x):
        # x: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames = x.shape[:2]
        
        x = x.view(-1, *x.shape[2:])
        features = self.resnet(x)
        
        features = features.view(batch_size, num_frames, -1)
        
        # Temporal attention pooling
        attention_weights = self.temporal_attention(features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        pooled_features = (features * attention_weights).sum(dim=1)
        
        return self.projection(pooled_features)


class AudioEncoder(nn.Module):
    """Audio encoder using 2D CNN on mel-spectrograms"""
    
    def __init__(self, output_dim=256, dropout=0.2):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Block 4
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
        
    def forward(self, x):
        # x: [batch_size, 1, n_mels, time_frames]
        
        for block in self.conv_blocks:
            x = block(x)
        
        x = x.squeeze(-1).squeeze(-1)
        
        return self.projection(x)


class GatedMultimodalUnit(nn.Module):
    """GMU for multimodal fusion"""
    
    def __init__(self, input_dim=256, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates
        self.gate_text = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        self.gate_video = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        self.gate_audio = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        # Transformations
        self.transform_text = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.transform_video = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.transform_audio = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_feat, video_feat, audio_feat):
        concat_feat = torch.cat([text_feat, video_feat, audio_feat], dim=-1)
        
        # Compute gates
        g_text = self.gate_text(concat_feat)
        g_video = self.gate_video(concat_feat)
        g_audio = self.gate_audio(concat_feat)
        
        # Transform
        h_text = self.transform_text(text_feat)
        h_video = self.transform_video(video_feat)
        h_audio = self.transform_audio(audio_feat)
        
        # Apply gates
        h_text = g_text * h_text
        h_video = g_video * h_video
        h_audio = g_audio * h_audio
        
        # Fuse
        fused = h_text + h_video + h_audio
        output = self.fusion(fused)
        
        return output, {
            'text': g_text.mean(dim=-1),
            'video': g_video.mean(dim=-1),
            'audio': g_audio.mean(dim=-1)
        }


class MultimodalEmotionModelV1(nn.Module):
    """Multimodal model for emotion and sentiment classification"""
    
    def __init__(self, 
                 num_emotions=7, 
                 num_sentiments=3,
                 feature_dim=256,
                 fusion_dim=256,
                 dropout=0.2,
                 analysis_mode=False):
        super().__init__()
        
        self.version = "Version 1"
        self.config = {
            'text_unfrozen_layers': 2,
            'video_unfrozen_strategy': 'layer4',
            'dropout': dropout,
            'feature_dim': feature_dim,
            'fusion_dim': fusion_dim,
            'learning_rate': 0.0002,
            'batch_size': 32
        }
        
        self.analysis_mode = analysis_mode
        self.num_emotions = num_emotions
        self.num_sentiments = num_sentiments
        
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        if self.analysis_mode:
            self.analysis_dir = Path("analysis_model_v1")
            self.analysis_dir.mkdir(exist_ok=True)
            self.training_history = {
                'train_loss': [], 'val_loss': [],
                'train_emotion_f1': [], 'val_emotion_f1': [],
                'train_sentiment_f1': [], 'val_sentiment_f1': [],
                'learning_rates': [], 'epoch_times': [],
                'gpu_memory': [], 'cpu_usage': [],
                'gate_weights_history': []
            }
        
        # Model components
        self.text_encoder = TextEncoder(output_dim=feature_dim, dropout=dropout)
        self.video_encoder = VideoEncoder(output_dim=feature_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(output_dim=feature_dim, dropout=dropout)
        
        self.fusion = GatedMultimodalUnit(
            input_dim=feature_dim,
            hidden_dim=fusion_dim,
            dropout=dropout
        )
        
        # Classifiers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_emotions)
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_sentiments)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, text_inputs, video_frames, audio_features):
        # Encode modalities
        text_feat = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        video_feat = self.video_encoder(video_frames)
        audio_feat = self.audio_encoder(audio_features)
        
        # Fuse
        fused_feat, gate_weights = self.fusion(text_feat, video_feat, audio_feat)
        
        # Classify
        emotion_logits = self.emotion_classifier(fused_feat)
        sentiment_logits = self.sentiment_classifier(fused_feat)
        
        return {
            'emotion_logits': emotion_logits,
            'sentiment_logits': sentiment_logits,
            'gate_weights': gate_weights,
            'features': {
                'text': text_feat,
                'video': video_feat,
                'audio': audio_feat,
                'fused': fused_feat
            }
        }
    
    def get_num_parameters(self):
        """Get parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        text_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        video_params = sum(p.numel() for p in self.video_encoder.parameters() if p.requires_grad)
        audio_params = sum(p.numel() for p in self.audio_encoder.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        classifier_params = trainable_params - text_params - video_params - audio_params - fusion_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'text_encoder': text_params,
            'video_encoder': video_params,
            'audio_encoder': audio_params,
            'fusion': fusion_params,
            'classifiers': classifier_params
        }
    
    def log_training_metrics(self, epoch, train_loss, val_loss, train_emotion_f1, val_emotion_f1,
                           train_sentiment_f1, val_sentiment_f1, learning_rate, epoch_time, gate_weights):
        """Log training metrics"""
        if not self.analysis_mode:
            return
        
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_emotion_f1'].append(train_emotion_f1)
        self.training_history['val_emotion_f1'].append(val_emotion_f1)
        self.training_history['train_sentiment_f1'].append(train_sentiment_f1)
        self.training_history['val_sentiment_f1'].append(val_sentiment_f1)
        self.training_history['learning_rates'].append(learning_rate)
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['gate_weights_history'].append(gate_weights)
        
        # System resources
        try:
            if GPU_AVAILABLE:
                gpu_info = GPUtil.getGPUs()
                if gpu_info:
                    gpu_memory = gpu_info[0].memoryUsed / gpu_info[0].memoryTotal * 100
                    self.training_history['gpu_memory'].append(gpu_memory)
                else:
                    self.training_history['gpu_memory'].append(0)
            else:
                self.training_history['gpu_memory'].append(0)
            
            cpu_usage = psutil.cpu_percent()
            self.training_history['cpu_usage'].append(cpu_usage)
        except:
            self.training_history['gpu_memory'].append(0)
            self.training_history['cpu_usage'].append(0)


def create_model_v1(num_emotions=7, num_sentiments=3, analysis_mode=False, device='cuda'):
    """Create Version 1 model"""
    model = MultimodalEmotionModelV1(
        num_emotions=num_emotions,
        num_sentiments=num_sentiments,
        analysis_mode=analysis_mode
    )
    
    params = model.get_num_parameters()
    print(f"\nModel V1 Configuration:")
    print(f"  Text Layers Unfrozen: 2 (layers 4-5)")
    print(f"  Video Strategy: layer4 only")
    print(f"  Learning Rate: 0.0002")
    print(f"  Batch Size: 32")
    print(f"  Dropout: 0.2")
    
    print(f"\nModel V1 Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  - Text Encoder: {params['text_encoder']:,}")
    print(f"  - Video Encoder: {params['video_encoder']:,}")
    print(f"  - Audio Encoder: {params['audio_encoder']:,}")
    print(f"  - Fusion: {params['fusion']:,}")
    print(f"  - Classifiers: {params['classifiers']:,}")
    
    if analysis_mode:
        print(f"  Analysis mode enabled")
    
    return model.to(device)


if __name__ == "__main__":
    print("Testing Model V1...")
    
    model = create_model_v1(analysis_mode=True)
    
    # Test inputs
    batch_size = 4
    text_inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    video_frames = torch.randn(batch_size, 30, 3, 224, 224)
    audio_features = torch.randn(batch_size, 1, 64, 300)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    video_frames = video_frames.to(device)
    audio_features = audio_features.to(device)
    
    # Forward pass
    outputs = model(text_inputs, video_frames, audio_features)
    
    print(f"\nModel V1 Output Shapes:")
    print(f"  Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"  Sentiment logits: {outputs['sentiment_logits'].shape}")
    
    print(f"\nGate weights (modality importance):")
    for modality, weight in outputs['gate_weights'].items():
        print(f"  {modality}: {weight.mean().item():.3f}")
    
    print(f"\nModel V1 testing complete!")