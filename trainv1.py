import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                           classification_report, roc_curve, auc, 
                           precision_recall_curve, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
import argparse
import json
import time
import psutil
from pathlib import Path
import warnings
from collections import defaultdict

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from modelv1 import create_model_v1, MultimodalEmotionModelV1
from meld_dataset import create_data_loaders

warnings.filterwarnings('ignore')


class EnhancedMultimodalTrainer:
    """Trainer for multimodal emotion model"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config, version_name="V1"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.version_name = version_name
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # A100 optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        self.setup_comprehensive_metrics()
        
        self.optimizer = self._create_optimizer()
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.scaler = GradScaler()
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.current_epoch = 0
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'runs/{version_name.lower()}_training_{timestamp}')
        
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
    def setup_comprehensive_metrics(self):
        """Initialize metrics tracking"""
        self.analysis_dir = Path(f"results_analysis_{self.version_name.lower()}")
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.training_metrics = {
            'epochs': [],
            'train_loss': [], 'val_loss': [],
            'train_emotion_loss': [], 'val_emotion_loss': [],
            'train_sentiment_loss': [], 'val_sentiment_loss': [],
            'learning_rates': [],
            'epoch_times': [],
            'convergence_metrics': [],
            'early_stopping_epoch': None
        }
        
        metric_names = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'precision_macro', 'recall_weighted', 'recall_macro']
        
        self.performance_metrics = {
            'emotion_metrics': {},
            'sentiment_metrics': {}
        }
        
        for task in ['emotion_metrics', 'sentiment_metrics']:
            self.performance_metrics[task] = {'class_wise_metrics': {}}
            for metric in metric_names:
                self.performance_metrics[task][f'train_{metric}'] = []
                self.performance_metrics[task][f'val_{metric}'] = []
                self.performance_metrics[task][f'test_{metric}'] = None
        
        self.comparison_metrics = {
            'version': self.version_name,
            'model_parameters': {},
            'final_performance': {},
            'training_efficiency': {}
        }
        
        self.modality_metrics = {
            'gate_weights_history': [],
            'modality_contributions': [],
            'fusion_effectiveness': {}
        }
        
        self.computational_metrics = {
            'gpu_memory_usage': [],
            'cpu_usage': [],
            'training_throughput': [],
            'inference_speed': None,
            'memory_peak': None
        }
        
        self.stored_predictions = {
            'train_emotion_probs': [], 'train_emotion_labels': [], 'train_emotion_preds': [],
            'train_sentiment_probs': [], 'train_sentiment_labels': [], 'train_sentiment_preds': [],
            'val_emotion_probs': [], 'val_emotion_labels': [], 'val_emotion_preds': [],
            'val_sentiment_probs': [], 'val_sentiment_labels': [], 'val_sentiment_preds': [],
            'test_emotion_probs': [], 'test_emotion_labels': [], 'test_emotion_preds': [],
            'test_sentiment_probs': [], 'test_sentiment_labels': [], 'test_sentiment_preds': []
        }
    
    def _create_optimizer(self):
        """Create optimizer with component-specific learning rates"""
        base_lr = self.config['learning_rate']
        
        param_groups = [
            {'params': self.model.text_encoder.distilbert.parameters(), 'lr': base_lr * 0.01, 'weight_decay': 0.01},
            {'params': self.model.video_encoder.resnet.parameters(), 'lr': base_lr * 0.01, 'weight_decay': 0.01},
            {'params': self.model.text_encoder.projection.parameters(), 'lr': base_lr * 0.1, 'weight_decay': 0.001},
            {'params': self.model.video_encoder.projection.parameters(), 'lr': base_lr * 0.1, 'weight_decay': 0.001},
            {'params': self.model.audio_encoder.parameters(), 'lr': base_lr * 0.5, 'weight_decay': 0.001},
            {'params': self.model.fusion.parameters(), 'lr': base_lr, 'weight_decay': 0.0001},
            {'params': self.model.emotion_classifier.parameters(), 'lr': base_lr, 'weight_decay': 0.0001},
            {'params': self.model.sentiment_classifier.parameters(), 'lr': base_lr, 'weight_decay': 0.0001}
        ]
        
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    def calculate_comprehensive_metrics(self, emotion_labels, emotion_preds, 
                                      sentiment_labels, sentiment_preds, 
                                      emotion_probs, sentiment_probs, phase='train'):
        """Calculate all metrics"""
        
        emotion_metrics = {
            'accuracy': accuracy_score(emotion_labels, emotion_preds),
            'f1_weighted': f1_score(emotion_labels, emotion_preds, average='weighted'),
            'f1_macro': f1_score(emotion_labels, emotion_preds, average='macro'),
            'precision_weighted': precision_score(emotion_labels, emotion_preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(emotion_labels, emotion_preds, average='macro', zero_division=0),
            'recall_weighted': recall_score(emotion_labels, emotion_preds, average='weighted', zero_division=0),
            'recall_macro': recall_score(emotion_labels, emotion_preds, average='macro', zero_division=0)
        }
        
        emotion_metrics['class_wise'] = {
            'f1_scores': f1_score(emotion_labels, emotion_preds, average=None, zero_division=0).tolist(),
            'precision_scores': precision_score(emotion_labels, emotion_preds, average=None, zero_division=0).tolist(),
            'recall_scores': recall_score(emotion_labels, emotion_preds, average=None, zero_division=0).tolist(),
            'support': np.bincount(emotion_labels, minlength=7).tolist()
        }
        
        sentiment_metrics = {
            'accuracy': accuracy_score(sentiment_labels, sentiment_preds),
            'f1_weighted': f1_score(sentiment_labels, sentiment_preds, average='weighted'),
            'f1_macro': f1_score(sentiment_labels, sentiment_preds, average='macro'),
            'precision_weighted': precision_score(sentiment_labels, sentiment_preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(sentiment_labels, sentiment_preds, average='macro', zero_division=0),
            'recall_weighted': recall_score(sentiment_labels, sentiment_preds, average='weighted', zero_division=0),
            'recall_macro': recall_score(sentiment_labels, sentiment_preds, average='macro', zero_division=0)
        }
        
        sentiment_metrics['class_wise'] = {
            'f1_scores': f1_score(sentiment_labels, sentiment_preds, average=None, zero_division=0).tolist(),
            'precision_scores': precision_score(sentiment_labels, sentiment_preds, average=None, zero_division=0).tolist(),
            'recall_scores': recall_score(sentiment_labels, sentiment_preds, average=None, zero_division=0).tolist(),
            'support': np.bincount(sentiment_labels, minlength=3).tolist()
        }
        
        return emotion_metrics, sentiment_metrics
    
    def _log_computational_metrics(self):
        """Log resource metrics"""
        try:
            cpu_usage = psutil.cpu_percent()
            self.computational_metrics['cpu_usage'].append(cpu_usage)
            
            if GPU_AVAILABLE and torch.cuda.is_available():
                gpu_info = GPUtil.getGPUs()
                if gpu_info:
                    gpu_memory_used = gpu_info[0].memoryUsed
                    gpu_memory_total = gpu_info[0].memoryTotal
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                    self.computational_metrics['gpu_memory_usage'].append(gpu_memory_percent)
                    
                    if self.computational_metrics['memory_peak'] is None:
                        self.computational_metrics['memory_peak'] = gpu_memory_used
                    else:
                        self.computational_metrics['memory_peak'] = max(
                            self.computational_metrics['memory_peak'], gpu_memory_used
                        )
                else:
                    self.computational_metrics['gpu_memory_usage'].append(0)
            else:
                self.computational_metrics['gpu_memory_usage'].append(0)
                
        except Exception as e:
            self.computational_metrics['cpu_usage'].append(0)
            self.computational_metrics['gpu_memory_usage'].append(0)
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        epoch_start_time = time.time()
        self.model.train()
        
        total_loss = 0
        total_emotion_loss = 0
        total_sentiment_loss = 0
        emotion_preds, emotion_labels = [], []
        sentiment_preds, sentiment_labels = [], []
        emotion_probs, sentiment_probs = [], []
        gate_weights_epoch = []
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            text_inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch['text_inputs'].items()}
            video_frames = batch['video_frames'].to(self.device, non_blocking=True)
            audio_features = batch['audio_features'].to(self.device, non_blocking=True)
            emotion_targets = batch['emotion_label'].to(self.device, non_blocking=True)
            sentiment_targets = batch['sentiment_label'].to(self.device, non_blocking=True)
            
            with autocast():
                outputs = self.model(text_inputs, video_frames, audio_features)
                
                emotion_loss = self.emotion_criterion(outputs['emotion_logits'], emotion_targets)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment_logits'], sentiment_targets)
                loss = emotion_loss + sentiment_loss
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * self.accumulation_steps
            total_emotion_loss += emotion_loss.item()
            total_sentiment_loss += sentiment_loss.item()
            
            emotion_batch_preds = outputs['emotion_logits'].argmax(dim=1).detach().cpu().numpy()
            sentiment_batch_preds = outputs['sentiment_logits'].argmax(dim=1).detach().cpu().numpy()
            emotion_batch_probs = torch.softmax(outputs['emotion_logits'], dim=1).detach().cpu().numpy()
            sentiment_batch_probs = torch.softmax(outputs['sentiment_logits'], dim=1).detach().cpu().numpy()
            
            emotion_preds.extend(emotion_batch_preds)
            emotion_labels.extend(emotion_targets.detach().cpu().numpy())
            sentiment_preds.extend(sentiment_batch_preds)
            sentiment_labels.extend(sentiment_targets.detach().cpu().numpy())
            emotion_probs.extend(emotion_batch_probs)
            sentiment_probs.extend(sentiment_batch_probs)
            
            if 'gate_weights' in outputs and outputs['gate_weights'] is not None:
                gate_weights_epoch.append({
                    'text': outputs['gate_weights']['text'].mean().detach().cpu().item(),
                    'video': outputs['gate_weights']['video'].mean().detach().cpu().item(),
                    'audio': outputs['gate_weights']['audio'].mean().detach().cpu().item()
                })
            
            pbar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'emotion_loss': f'{emotion_loss.item():.4f}',
                'sentiment_loss': f'{sentiment_loss.item():.4f}'
            })
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_loader)
        avg_emotion_loss = total_emotion_loss / len(self.train_loader)
        avg_sentiment_loss = total_sentiment_loss / len(self.train_loader)
        
        if gate_weights_epoch:
            avg_gates = {
                'text': np.mean([gw['text'] for gw in gate_weights_epoch]),
                'video': np.mean([gw['video'] for gw in gate_weights_epoch]),
                'audio': np.mean([gw['audio'] for gw in gate_weights_epoch])
            }
            self.modality_metrics['gate_weights_history'].append(avg_gates)
        
        emotion_metrics, sentiment_metrics = self.calculate_comprehensive_metrics(
            emotion_labels, emotion_preds, sentiment_labels, sentiment_preds,
            emotion_probs, sentiment_probs, phase='train'
        )
        
        self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'emotion_loss': avg_emotion_loss,
            'sentiment_loss': avg_sentiment_loss,
            'emotion_f1': emotion_metrics['f1_weighted'],
            'sentiment_f1': sentiment_metrics['f1_weighted'],
            'emotion_acc': emotion_metrics['accuracy'],
            'sentiment_acc': sentiment_metrics['accuracy'],
            'epoch_time': epoch_time,
            'emotion_metrics': emotion_metrics,
            'sentiment_metrics': sentiment_metrics
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader, phase='val', store_predictions=False):
        """Evaluate model"""
        self.model.eval()
        
        total_loss = 0
        total_emotion_loss = 0
        total_sentiment_loss = 0
        emotion_preds, emotion_labels = [], []
        sentiment_preds, sentiment_labels = [], []
        emotion_probs, sentiment_probs = [], []
        
        for batch in tqdm(data_loader, desc=f'Evaluating {phase}'):
            if batch is None:
                continue
            
            text_inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch['text_inputs'].items()}
            video_frames = batch['video_frames'].to(self.device, non_blocking=True)
            audio_features = batch['audio_features'].to(self.device, non_blocking=True)
            emotion_targets = batch['emotion_label'].to(self.device, non_blocking=True)
            sentiment_targets = batch['sentiment_label'].to(self.device, non_blocking=True)
            
            with autocast():
                outputs = self.model(text_inputs, video_frames, audio_features)
                emotion_loss = self.emotion_criterion(outputs['emotion_logits'], emotion_targets)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment_logits'], sentiment_targets)
                loss = emotion_loss + sentiment_loss
            
            total_loss += loss.item()
            total_emotion_loss += emotion_loss.item()
            total_sentiment_loss += sentiment_loss.item()
            
            emotion_batch_preds = outputs['emotion_logits'].argmax(dim=1).cpu().numpy()
            sentiment_batch_preds = outputs['sentiment_logits'].argmax(dim=1).cpu().numpy()
            emotion_batch_probs = torch.softmax(outputs['emotion_logits'], dim=1).cpu().numpy()
            sentiment_batch_probs = torch.softmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
            
            emotion_preds.extend(emotion_batch_preds)
            emotion_labels.extend(emotion_targets.cpu().numpy())
            sentiment_preds.extend(sentiment_batch_preds)
            sentiment_labels.extend(sentiment_targets.cpu().numpy())
            emotion_probs.extend(emotion_batch_probs)
            sentiment_probs.extend(sentiment_batch_probs)
        
        avg_loss = total_loss / len(data_loader)
        avg_emotion_loss = total_emotion_loss / len(data_loader)
        avg_sentiment_loss = total_sentiment_loss / len(data_loader)
        
        if store_predictions:
            self.stored_predictions[f'{phase}_emotion_probs'] = np.array(emotion_probs)
            self.stored_predictions[f'{phase}_emotion_labels'] = np.array(emotion_labels)
            self.stored_predictions[f'{phase}_emotion_preds'] = np.array(emotion_preds)
            self.stored_predictions[f'{phase}_sentiment_probs'] = np.array(sentiment_probs)
            self.stored_predictions[f'{phase}_sentiment_labels'] = np.array(sentiment_labels)
            self.stored_predictions[f'{phase}_sentiment_preds'] = np.array(sentiment_preds)
        
        emotion_metrics, sentiment_metrics = self.calculate_comprehensive_metrics(
            emotion_labels, emotion_preds, sentiment_labels, sentiment_preds,
            emotion_probs, sentiment_probs, phase=phase
        )
        
        return {
            'loss': avg_loss,
            'emotion_loss': avg_emotion_loss,
            'sentiment_loss': avg_sentiment_loss,
            'emotion_acc': emotion_metrics['accuracy'],
            'emotion_f1': emotion_metrics['f1_weighted'],
            'sentiment_acc': sentiment_metrics['accuracy'],
            'sentiment_f1': sentiment_metrics['f1_weighted'],
            'emotion_metrics': emotion_metrics,
            'sentiment_metrics': sentiment_metrics
        }
    
    def log_epoch_metrics(self, epoch, train_metrics, val_metrics, epoch_time):
        """Log epoch metrics"""
        
        self.training_metrics['epochs'].append(epoch + 1)
        self.training_metrics['train_loss'].append(train_metrics['loss'])
        self.training_metrics['val_loss'].append(val_metrics['loss'])
        self.training_metrics['train_emotion_loss'].append(train_metrics.get('emotion_loss', 0))
        self.training_metrics['val_emotion_loss'].append(val_metrics.get('emotion_loss', 0))
        self.training_metrics['train_sentiment_loss'].append(train_metrics.get('sentiment_loss', 0))
        self.training_metrics['val_sentiment_loss'].append(val_metrics.get('sentiment_loss', 0))
        self.training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        self.training_metrics['epoch_times'].append(epoch_time)
        
        val_combined_f1 = (val_metrics['emotion_f1'] + val_metrics['sentiment_f1']) / 2
        self.training_metrics['convergence_metrics'].append({
            'epoch': epoch + 1,
            'val_combined_f1': val_combined_f1,
            'improvement': val_combined_f1 > self.best_val_f1,
            'patience_counter': self.patience_counter
        })
        
        for metric_name in ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'precision_macro', 'recall_weighted', 'recall_macro']:
            self.performance_metrics['emotion_metrics'][f'train_{metric_name}'].append(train_metrics['emotion_metrics'][metric_name])
            self.performance_metrics['emotion_metrics'][f'val_{metric_name}'].append(val_metrics['emotion_metrics'][metric_name])
            self.performance_metrics['sentiment_metrics'][f'train_{metric_name}'].append(train_metrics['sentiment_metrics'][metric_name])
            self.performance_metrics['sentiment_metrics'][f'val_{metric_name}'].append(val_metrics['sentiment_metrics'][metric_name])
        
        self._log_computational_metrics()
        
        self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        self.writer.add_scalar('train/emotion_f1', train_metrics['emotion_f1'], epoch)
        self.writer.add_scalar('val/emotion_f1', val_metrics['emotion_f1'], epoch)
        self.writer.add_scalar('train/sentiment_f1', train_metrics['sentiment_f1'], epoch)
        self.writer.add_scalar('val/sentiment_f1', val_metrics['sentiment_f1'], epoch)
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('epoch_time', epoch_time, epoch)
    
    def calculate_model_parameters(self):
        """Calculate parameter counts"""
        if hasattr(self.model, 'get_num_parameters'):
            params = self.model.get_num_parameters()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            params = {'total': total_params, 'trainable': trainable_params}
        
        self.comparison_metrics['model_parameters'].update(params)
        return params
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting {self.version_name} training")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        params = self.calculate_model_parameters()
        print(f"Model Parameters - Total: {params.get('total', 0):,}, Trainable: {params.get('trainable', 0):,}")
        
        training_start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, 'val', store_predictions=True)
            
            self.log_epoch_metrics(epoch, train_metrics, val_metrics, train_metrics['epoch_time'])
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Emotion F1: {train_metrics['emotion_f1']:.4f}, "
                  f"Sentiment F1: {train_metrics['sentiment_f1']:.4f}")
            print(f"  Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Emotion F1: {val_metrics['emotion_f1']:.4f}, "
                  f"Sentiment F1: {val_metrics['sentiment_f1']:.4f}")
            
            combined_f1 = (val_metrics['emotion_f1'] + val_metrics['sentiment_f1']) / 2
            if combined_f1 > self.best_val_f1:
                self.best_val_f1 = combined_f1
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered")
                self.training_metrics['early_stopping_epoch'] = epoch + 1
                break
        
        print(f"\nRunning final test evaluation")
        test_metrics = self.evaluate(self.test_loader, 'test', store_predictions=True)
        
        for metric_name in ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'precision_macro', 'recall_weighted', 'recall_macro']:
            self.performance_metrics['emotion_metrics'][f'test_{metric_name}'] = test_metrics['emotion_metrics'][metric_name]
            self.performance_metrics['sentiment_metrics'][f'test_{metric_name}'] = test_metrics['sentiment_metrics'][metric_name]
        
        self.performance_metrics['emotion_metrics']['class_wise_metrics'] = test_metrics['emotion_metrics']['class_wise']
        self.performance_metrics['sentiment_metrics']['class_wise_metrics'] = test_metrics['sentiment_metrics']['class_wise']
        
        total_training_time = time.time() - training_start_time
        self.comparison_metrics['training_efficiency'] = {
            'total_training_time_hours': total_training_time / 3600,
            'average_epoch_time_minutes': np.mean(self.training_metrics['epoch_times']) / 60,
            'samples_per_second': len(self.train_loader.dataset) * len(self.training_metrics['epochs']) / total_training_time,
            'early_stopping_epoch': self.training_metrics['early_stopping_epoch'],
            'convergence_efficiency': len(self.training_metrics['epochs']) / self.config['num_epochs']
        }
        
        self.comparison_metrics['final_performance'] = {
            'test_emotion_f1': test_metrics['emotion_f1'],
            'test_sentiment_f1': test_metrics['sentiment_f1'],
            'test_combined_f1': (test_metrics['emotion_f1'] + test_metrics['sentiment_f1']) / 2,
            'best_val_combined_f1': self.best_val_f1
        }
        
        print(f"\nTraining Complete")
        print(f"Total Training Time: {total_training_time/3600:.2f} hours")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"Test Results:")
        print(f"  Emotion F1: {test_metrics['emotion_f1']:.4f}")
        print(f"  Sentiment F1: {test_metrics['sentiment_f1']:.4f}")
        
        self.save_comprehensive_metrics()
        
        self.writer.close()
        return test_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'performance_metrics': self.performance_metrics,
            'comparison_metrics': self.comparison_metrics,
            'modality_metrics': self.modality_metrics,
            'computational_metrics': self.computational_metrics
        }
        
        checkpoint_dir = Path(self.config['save_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / f'best_model_{self.version_name.lower()}.pt'
            torch.save(checkpoint, best_path)
    
    def save_comprehensive_metrics(self):
        """Save all metrics"""
        
        comprehensive_results = {
            'version': self.version_name,
            'training_dynamics': self.training_metrics,
            'performance_evaluation': self.performance_metrics,
            'cross_version_comparison': self.comparison_metrics,
            'modality_analysis': self.modality_metrics,
            'computational_analysis': self.computational_metrics,
            'stored_predictions': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in self.stored_predictions.items()},
            'config': self.config
        }
        
        results_file = self.analysis_dir / f'{self.version_name.lower()}_comprehensive_results.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train V1 multimodal emotion model')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--version', type=str, default='V1', help='Model version identifier')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['analysis_mode'] = True
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"Loading MELD dataset")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv=config['train_csv'],
        train_video_dir=config['train_video_dir'],
        val_csv=config['val_csv'],
        val_video_dir=config['val_video_dir'],
        test_csv=config['test_csv'],
        test_video_dir=config['test_video_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        **config.get('data_params', {})
    )
    
    print(f"Creating {args.version} model")
    
    model = create_model_v1(
        num_emotions=config['num_emotions'],
        num_sentiments=config['num_sentiments'],
        analysis_mode=True
    )
    
    trainer = EnhancedMultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        version_name=args.version
    )
    
    final_results = trainer.train()


if __name__ == "__main__":
    main()