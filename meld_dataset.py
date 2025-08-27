import os
import cv2
import torch
import numpy as np
import pandas as pd
import subprocess
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
import json
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MELDDatasetV1(Dataset):
    """MELD Dataset for multimodal emotion and sentiment analysis"""
    
    def __init__(self, csv_path, video_dir, 
                 max_frames=30, 
                 frame_sample_rate=3,
                 max_text_length=128,
                 audio_sample_rate=16000,
                 n_mels=64,
                 analysis_mode=False):
        
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.max_frames = max_frames
        self.frame_sample_rate = frame_sample_rate
        self.max_text_length = max_text_length
        self.audio_sample_rate = audio_sample_rate
        self.n_mels = n_mels
        self.analysis_mode = analysis_mode
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Label mappings
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }
        
        self.emotion_labels = list(self.emotion_map.keys())
        self.sentiment_labels = list(self.sentiment_map.keys())
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512,
            f_min=0,
            f_max=8000
        )
        
        self.video_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.video_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        print(f"Loaded MELD Dataset with {len(self.data)} samples")
        
        if self.analysis_mode:
            self.perform_comprehensive_analysis()
    
    def perform_comprehensive_analysis(self):
        """Run all analysis functions"""
        print("Performing comprehensive data analysis...")
        
        self.analysis_dir = Path("analysis_results")
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.analyze_class_distributions()
        self.analyze_text_features()
        self.analyze_dialogue_statistics()
        self.create_correlation_analysis()
        self.analyze_data_quality()
        
        print(f"Analysis complete. Results saved to {self.analysis_dir}")
    
    def analyze_class_distributions(self):
        """Analyze target class distributions"""
        emotion_counts = self.data['Emotion'].value_counts()
        sentiment_counts = self.data['Sentiment'].value_counts()
        
        # Color schemes
        emotion_colors = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
        sentiment_colors = ['#CC3311', '#BBBBBB', '#228833']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MELD Dataset - Target Class Distributions', fontsize=18, fontweight='bold', y=0.98)
        
        # Emotion bar plot
        ax1 = axes[0, 0]
        bars1 = ax1.bar(emotion_counts.index, emotion_counts.values, 
                       color=emotion_colors[:len(emotion_counts)], 
                       alpha=0.9, edgecolor='black', linewidth=1.2)
        ax1.set_title('Emotion Class Distribution', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Emotion Categories', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=11)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(['Sample Count'], loc='upper right', fontsize=10)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Sentiment bar plot
        ax2 = axes[0, 1]
        bars2 = ax2.bar(sentiment_counts.index, sentiment_counts.values, 
                       color=sentiment_colors[:len(sentiment_counts)], 
                       alpha=0.9, edgecolor='black', linewidth=1.2)
        ax2.set_title('Sentiment Class Distribution', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Sentiment Categories', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=11)
        ax2.tick_params(axis='y', labelsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(['Sample Count'], loc='upper right', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Emotion pie chart
        ax3 = axes[1, 0]
        wedges, texts = ax3.pie(emotion_counts.values, startangle=90, colors=emotion_colors[:len(emotion_counts)],
                               wedgeprops=dict(width=0.9, edgecolor='white', linewidth=2))
        
        ax3.set_title('Emotion Distribution', fontsize=14, fontweight='bold', pad=15)
        
        total_emotions = sum(emotion_counts.values)
        legend_labels = [f'{label} ({count}, {count/total_emotions*100:.1f}%)' 
                        for label, count in zip(emotion_counts.index, emotion_counts.values)]
        ax3.legend(wedges, legend_labels, title="Emotions", loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9, title_fontsize=10)
        
        # Sentiment pie chart
        ax4 = axes[1, 1]
        wedges2, texts2 = ax4.pie(sentiment_counts.values, startangle=90, colors=sentiment_colors[:len(sentiment_counts)],
                                 wedgeprops=dict(width=0.9, edgecolor='white', linewidth=2))
        
        ax4.set_title('Sentiment Distribution', fontsize=14, fontweight='bold', pad=15)
        
        total_sentiments = sum(sentiment_counts.values)
        legend_labels2 = [f'{label} ({count}, {count/total_sentiments*100:.1f}%)' 
                         for label, count in zip(sentiment_counts.index, sentiment_counts.values)]
        ax4.legend(wedges2, legend_labels2, title="Sentiments", loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9, title_fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.82, top=0.94)
        plt.savefig(self.analysis_dir / 'class_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        distribution_stats = {
            'emotion_distribution': emotion_counts.to_dict(),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'total_samples': len(self.data),
            'emotion_balance_ratio': emotion_counts.min() / emotion_counts.max(),
            'sentiment_balance_ratio': sentiment_counts.min() / sentiment_counts.max()
        }
        
        with open(self.analysis_dir / 'distribution_stats.json', 'w') as f:
            json.dump(distribution_stats, f, indent=2)
    
    def analyze_text_features(self):
        """Analyze text characteristics"""
        text_lengths = self.data['Utterance'].str.len()
        word_counts = self.data['Utterance'].str.split().str.len()
        
        primary_color = '#4477AA'
        accent_colors = ['#EE6677', '#CCBB44']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MELD Dataset - Text Feature Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Text length histogram
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(text_lengths, bins=50, color=primary_color, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Distribution of Text Lengths', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Number of Characters', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.axvline(text_lengths.mean(), color=accent_colors[0], linestyle='--', linewidth=2, 
                   label=f'Mean: {text_lengths.mean():.1f}')
        ax1.axvline(text_lengths.median(), color=accent_colors[1], linestyle='--', linewidth=2, 
                   label=f'Median: {text_lengths.median():.1f}')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
        
        # Word count histogram
        ax2 = axes[0, 1]
        n2, bins2, patches2 = ax2.hist(word_counts, bins=30, color='#228833', alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Distribution of Word Counts', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Number of Words', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.axvline(word_counts.mean(), color=accent_colors[0], linestyle='--', linewidth=2, 
                   label=f'Mean: {word_counts.mean():.1f}')
        ax2.axvline(word_counts.median(), color=accent_colors[1], linestyle='--', linewidth=2, 
                   label=f'Median: {word_counts.median():.1f}')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        
        # Text length by emotion boxplot
        ax3 = axes[1, 0]
        emotion_text_lengths = []
        emotions = []
        for emotion in self.emotion_labels:
            emotion_data = self.data[self.data['Emotion'] == emotion]['Utterance'].str.len()
            emotion_text_lengths.extend(emotion_data.tolist())
            emotions.extend([emotion] * len(emotion_data))
        
        emotion_df = pd.DataFrame({'emotion': emotions, 'text_length': emotion_text_lengths})
        
        emotion_colors = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
        bp = ax3.boxplot([emotion_df[emotion_df['emotion'] == emotion]['text_length'].values 
                         for emotion in self.emotion_labels], 
                        labels=self.emotion_labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], emotion_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax3.set_title('Text Length by Emotion', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Emotion Categories', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Text Length (Characters)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        ax3.tick_params(axis='y', labelsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Word count by sentiment boxplot
        ax4 = axes[1, 1]
        sentiment_word_counts = []
        sentiments = []
        for sentiment in self.sentiment_labels:
            sentiment_data = self.data[self.data['Sentiment'] == sentiment]['Utterance'].str.split().str.len()
            sentiment_word_counts.extend(sentiment_data.tolist())
            sentiments.extend([sentiment] * len(sentiment_data))
        
        sentiment_df = pd.DataFrame({'sentiment': sentiments, 'word_count': sentiment_word_counts})
        
        sentiment_colors = ['#CC3311', '#BBBBBB', '#228833']
        bp2 = ax4.boxplot([sentiment_df[sentiment_df['sentiment'] == sentiment]['word_count'].values 
                          for sentiment in self.sentiment_labels], 
                         labels=self.sentiment_labels, patch_artist=True)
        
        for patch, color in zip(bp2['boxes'], sentiment_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax4.set_title('Word Count by Sentiment', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Sentiment Categories', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Word Count', fontsize=12, fontweight='bold')
        ax4.tick_params(labelsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(self.analysis_dir / 'text_feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        text_stats = {
            'mean_text_length': float(text_lengths.mean()),
            'median_text_length': float(text_lengths.median()),
            'std_text_length': float(text_lengths.std()),
            'mean_word_count': float(word_counts.mean()),
            'median_word_count': float(word_counts.median()),
            'std_word_count': float(word_counts.std()),
            'max_text_length': int(text_lengths.max()),
            'min_text_length': int(text_lengths.min())
        }
        
        with open(self.analysis_dir / 'text_stats.json', 'w') as f:
            json.dump(text_stats, f, indent=2)
    
    def analyze_dialogue_statistics(self):
        """Analyze dialogue and speaker statistics"""
        dialogue_counts = self.data['Dialogue_ID'].value_counts()
        speaker_counts = self.data['Speaker'].value_counts() if 'Speaker' in self.data.columns else None
        
        primary_colors = ['#CCBB44', '#0077BB', '#EE7733', '#009988', '#EE3377']
        emotion_colors = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MELD Dataset - Dialogue and Speaker Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Utterances per dialogue distribution
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(dialogue_counts.values, bins=30, color=primary_colors[0], alpha=0.9, 
                                   edgecolor='black', linewidth=1)
        ax1.set_title('Distribution of Utterances per Dialogue', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Number of Utterances per Dialogue', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Dialogues', fontsize=12, fontweight='bold')
        ax1.axvline(dialogue_counts.mean(), color='#EE6677', linestyle='--', linewidth=2,
                   label=f'Mean: {dialogue_counts.mean():.1f}')
        ax1.axvline(dialogue_counts.median(), color='#228833', linestyle='--', linewidth=2,
                   label=f'Median: {dialogue_counts.median():.1f}')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
        
        # Dialogue length statistics
        ax2 = axes[0, 1]
        dialogue_stats = {
            'Min Length': dialogue_counts.min(),
            'Max Length': dialogue_counts.max(),
            'Mean Length': dialogue_counts.mean(),
            'Median Length': dialogue_counts.median(),
            'Std Dev': dialogue_counts.std()
        }
        
        bars = ax2.bar(dialogue_stats.keys(), dialogue_stats.values(), 
                      color=['#4477AA', '#EE6677', '#CCBB44', '#228833', '#AA3377'], 
                      alpha=0.9, edgecolor='black', linewidth=1)
        ax2.set_title('Dialogue Length Statistics', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Statistics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Utterances', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Speaker analysis
        if speaker_counts is not None:
            ax3 = axes[1, 0]
            top_speakers = speaker_counts.head(10)
            speaker_colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', 
                             '#DDCC77', '#CC6677', '#882255', '#AA4499', '#DDDDDD']
            bars = ax3.bar(range(len(top_speakers)), top_speakers.values, 
                          color=speaker_colors[:len(top_speakers)], 
                          alpha=0.9, edgecolor='black', linewidth=1)
            ax3.set_title('Top 10 Speakers by Utterance Count', fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlabel('Speakers', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Utterances', fontsize=12, fontweight='bold')
            ax3.set_xticks(range(len(top_speakers)))
            ax3.set_xticklabels(top_speakers.index, rotation=45, fontsize=10)
            ax3.tick_params(axis='y', labelsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax3 = axes[1, 0]
            ax3.text(0.5, 0.5, 'Speaker information not available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12, fontweight='bold')
            ax3.set_title('Speaker Analysis', fontsize=14, fontweight='bold', pad=15)
        
        # Emotion distribution across dialogues
        ax4 = axes[1, 1]
        dialogue_emotions = self.data.groupby('Dialogue_ID')['Emotion'].apply(lambda x: x.mode().iloc[0])
        emotion_dialogue_counts = dialogue_emotions.value_counts()
        
        available_colors = emotion_colors[:len(emotion_dialogue_counts)]
        
        wedges, texts = ax4.pie(emotion_dialogue_counts.values, startangle=90, colors=available_colors, 
                               wedgeprops=dict(width=0.9, edgecolor='white', linewidth=2))
        
        ax4.set_title('Dominant Emotion Across Dialogues', fontsize=14, fontweight='bold', pad=15)
        
        total_dialogues = sum(emotion_dialogue_counts.values)
        legend_labels = [f'{label} ({count}, {count/total_dialogues*100:.1f}%)' 
                        for label, count in zip(emotion_dialogue_counts.index, emotion_dialogue_counts.values)]
        ax4.legend(wedges, legend_labels, title="Emotions", loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9, title_fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.82, top=0.94)
        plt.savefig(self.analysis_dir / 'dialogue_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_analysis(self):
        """Create correlation heatmap"""
        numerical_features = []
        feature_names = []
        
        # Extract features
        text_lengths = self.data['Utterance'].str.len().values
        word_counts = self.data['Utterance'].str.split().str.len().values
        numerical_features.extend([text_lengths, word_counts])
        feature_names.extend(['Text Length', 'Word Count'])
        
        dialogue_lengths = self.data.groupby('Dialogue_ID').size()
        utterance_positions = self.data.groupby('Dialogue_ID').cumcount()
        numerical_features.extend([
            self.data['Dialogue_ID'].map(dialogue_lengths).values,
            utterance_positions.values
        ])
        feature_names.extend(['Dialogue Length', 'Utterance Position'])
        
        emotion_encoded = self.data['Emotion'].map(self.emotion_map).values
        sentiment_encoded = self.data['Sentiment'].map(self.sentiment_map).values
        numerical_features.extend([emotion_encoded, sentiment_encoded])
        feature_names.extend(['Emotion Label', 'Sentiment Label'])
        
        feature_matrix = np.array(numerical_features).T
        correlation_df = pd.DataFrame(feature_matrix, columns=feature_names)
        correlation_matrix = correlation_df.corr()
        
        plt.figure(figsize=(12, 10))
        
        # Custom colormap
        colors = ['#4477AA', '#66CCEE', '#FFFFFF', '#EE6677', '#CC3311']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
        
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0,
                   square=True, linewidths=1.5, cbar_kws={"shrink": .8},
                   annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                   fmt='.3f')
        
        plt.title('MELD Dataset - Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=14, fontweight='bold')
        plt.ylabel('Features', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        correlation_matrix.to_csv(self.analysis_dir / 'correlation_matrix.csv')
    
    def analyze_data_quality(self):
        """Analyze data quality"""
        missing_data = self.data.isnull().sum()
        duplicates = self.data.duplicated().sum()
        empty_texts = (self.data['Utterance'].str.strip() == '').sum()
        very_short_texts = (self.data['Utterance'].str.len() < 5).sum()
        
        quality_colors = ['#228833', '#EE6677']
        reference_colors = ['#EE6677', '#228833', '#CCBB44']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MELD Dataset - Data Quality Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Missing values plot
        ax1 = axes[0, 0]
        if missing_data.sum() > 0:
            bars = ax1.bar(missing_data.index, missing_data.values, color='#EE6677', alpha=0.8, 
                          edgecolor='black', linewidth=1)
            ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold', pad=15)
            ax1.set_ylabel('Number of Missing Values', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, fontweight='bold', color='#228833')
            ax1.set_title('Missing Values Analysis', fontsize=14, fontweight='bold', pad=15)
        
        # Text length distribution
        ax2 = axes[0, 1]
        text_lengths = self.data['Utterance'].str.len()
        n, bins, patches = ax2.hist(text_lengths, bins=50, color='#4477AA', alpha=0.8, edgecolor='black', linewidth=1)
        ax2.axvline(5, color=reference_colors[0], linestyle='--', linewidth=2, label='Very Short Threshold')
        ax2.axvline(text_lengths.median(), color=reference_colors[1], linestyle='--', linewidth=2, label='Median Length')
        ax2.axvline(text_lengths.mean(), color=reference_colors[2], linestyle='--', linewidth=2, label='Mean Length')
        ax2.set_title('Text Length with Quality Thresholds', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Text Length (Characters)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        
        # Data quality pie chart
        ax3 = axes[1, 0]
        quality_categories = ['Good Quality', 'Quality Issues']
        good_quality = len(self.data) - duplicates - empty_texts - very_short_texts
        quality_issues = duplicates + empty_texts + very_short_texts
        
        wedges, texts = ax3.pie([good_quality, quality_issues], startangle=90, colors=quality_colors,
                               wedgeprops=dict(width=0.9, edgecolor='white', linewidth=2))
        
        ax3.set_title('Overall Data Quality Assessment', fontsize=14, fontweight='bold', pad=15)
        
        total_samples = len(self.data)
        legend_labels = [f'{cat} ({val}, {val/total_samples*100:.1f}%)' 
                        for cat, val in zip(quality_categories, [good_quality, quality_issues])]
        ax3.legend(wedges, legend_labels, title="Quality Status", loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10, title_fontsize=10)
        
        # Data quality statistics
        ax4 = axes[1, 1]
        stats_data = [
            ('Total Samples', len(self.data)),
            ('Good Quality', good_quality),
            ('Duplicates', duplicates),
            ('Empty Texts', empty_texts),
            ('Very Short', very_short_texts)
        ]
        
        labels = [item[0] for item in stats_data]
        values = [item[1] for item in stats_data]
        colors = ['#4477AA', '#228833', '#EE6677', '#EE6677', '#CCBB44']
        
        bars = ax4.barh(labels, values, color=colors, alpha=0.9, edgecolor='black', linewidth=1)
        ax4.set_title('Data Quality Statistics', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Quality Metrics', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.tick_params(labelsize=10)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.82, top=0.94)
        plt.savefig(self.analysis_dir / 'data_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        quality_summary = {
            'total_samples': len(self.data),
            'missing_values': missing_data.to_dict(),
            'duplicate_entries': int(duplicates),
            'empty_texts': int(empty_texts),
            'very_short_texts': int(very_short_texts),
            'data_quality_score': float(1 - (duplicates + empty_texts + very_short_texts) / len(self.data))
        }
        
        with open(self.analysis_dir / 'data_quality_summary.json', 'w') as f:
            json.dump(quality_summary, f, indent=2)
        
        print(f"Data Quality Score: {quality_summary['data_quality_score']:.3f}")
    
    def get_analysis_summary(self):
        """Get analysis summary"""
        if not hasattr(self, 'analysis_dir'):
            print("Analysis not performed. Set analysis_mode=True during initialization.")
            return None
        
        summary = {
            'dataset_version': 'Version 1',
            'total_samples': len(self.data),
            'emotion_classes': len(self.emotion_labels),
            'sentiment_classes': len(self.sentiment_labels),
            'analysis_files': list(self.analysis_dir.glob('*.png')) + list(self.analysis_dir.glob('*.json')),
            'modalities': ['text', 'audio', 'video'],
            'preprocessing_parameters': {
                'max_frames': self.max_frames,
                'max_text_length': self.max_text_length,
                'audio_sample_rate': self.audio_sample_rate,
                'n_mels': self.n_mels
            }
        }
        
        return summary
    
    def __len__(self):
        return len(self.data)
    
    def _extract_video_frames(self, video_path):
        """Extract and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > self.max_frames:
            indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            if len(frames) >= self.max_frames:
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        frames = np.array(frames, dtype=np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        # Normalize
        frames = (frames - self.video_mean) / self.video_std
        
        # Pad if necessary
        if frames.size(0) < self.max_frames:
            padding = torch.zeros(self.max_frames - frames.size(0), 3, 224, 224)
            frames = torch.cat([frames, padding], dim=0)
        
        return frames[:self.max_frames]
    
    def _extract_audio_features(self, video_path):
        """Extract audio and convert to mel-spectrogram"""
        temp_audio = video_path.replace('.mp4', '_temp.wav')
        
        try:
            # Extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',
                '-y',
                temp_audio
            ]
            
            subprocess.run(cmd, check=True, 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
            
            waveform, sr = torchaudio.load(temp_audio)
            
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mel-spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Log-scale and normalize
            mel_spec = torch.log(mel_spec + 1e-9)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
            
            # Ensure consistent time dimension
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]
            
            return mel_spec
            
        except Exception as e:
            print(f"Audio extraction failed for {video_path}: {e}")
            return torch.zeros(1, self.n_mels, 300)
            
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def _tokenize_text(self, text):
        """Tokenize text"""
        text = str(text).strip()
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def __getitem__(self, idx):
        """Get a single sample"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data.iloc[idx]
        
        video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None
        
        try:
            video_frames = self._extract_video_frames(video_path)
            audio_features = self._extract_audio_features(video_path)
            text_encoding = self._tokenize_text(row['Utterance'])
            
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]
            
            return {
                'video_frames': video_frames,
                'audio_features': audio_features,
                'text_inputs': text_encoding,
                'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
                'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
                'dialogue_id': row['Dialogue_ID'],
                'utterance_id': row['Utterance_ID'],
                'raw_text': row['Utterance'],
                'emotion_name': row['Emotion'],
                'sentiment_name': row['Sentiment']
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return None


def collate_fn_v1(batch):
    """Collate function with metadata preservation"""
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    return {
        'video_frames': torch.stack([s['video_frames'] for s in batch]),
        'audio_features': torch.stack([s['audio_features'] for s in batch]),
        'text_inputs': {
            'input_ids': torch.stack([s['text_inputs']['input_ids'] for s in batch]),
            'attention_mask': torch.stack([s['text_inputs']['attention_mask'] for s in batch])
        },
        'emotion_label': torch.stack([s['emotion_label'] for s in batch]),
        'sentiment_label': torch.stack([s['sentiment_label'] for s in batch]),
        'metadata': {
            'dialogue_ids': [s['dialogue_id'] for s in batch],
            'utterance_ids': [s['utterance_id'] for s in batch],
            'raw_texts': [s['raw_text'] for s in batch],
            'emotion_names': [s['emotion_name'] for s in batch],
            'sentiment_names': [s['sentiment_name'] for s in batch]
        }
    }


def create_data_loaders_v1(train_csv, train_video_dir,
                          val_csv, val_video_dir,
                          test_csv, test_video_dir,
                          batch_size=32,
                          num_workers=4,
                          analysis_mode=False,
                          **dataset_kwargs):
    """Create data loaders with analysis capabilities"""
    
    train_dataset = MELDDatasetV1(train_csv, train_video_dir, 
                                  analysis_mode=analysis_mode, **dataset_kwargs)
    val_dataset = MELDDatasetV1(val_csv, val_video_dir, **dataset_kwargs)
    test_dataset = MELDDatasetV1(test_csv, test_video_dir, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_v1,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_v1,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_v1,
        pin_memory=True
    )
    
    print(f"MELD Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    analysis_summary = train_dataset.get_analysis_summary() if analysis_mode else None
    
    return train_loader, val_loader, test_loader, analysis_summary


def create_data_loaders(train_csv, train_video_dir, val_csv, val_video_dir,
                       test_csv, test_video_dir, batch_size=32, num_workers=4, **kwargs):
    """Compatibility wrapper"""
    train_loader, val_loader, test_loader, _ = create_data_loaders_v1(
        train_csv, train_video_dir, val_csv, val_video_dir,
        test_csv, test_video_dir, batch_size, num_workers, **kwargs
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing MELD Dataset with Analysis...")
    
    dataset = MELDDatasetV1(
        csv_path='../dataset/train/train_sent_emo.csv',
        video_dir='../dataset/train/train_splits',
        analysis_mode=True
    )
    
    # Test sample loading
    sample = dataset[0]
    if sample is not None:
        print("\nSample shapes and metadata:")
        print(f"  Video: {sample['video_frames'].shape}")
        print(f"  Audio: {sample['audio_features'].shape}")
        print(f"  Text input_ids: {sample['text_inputs']['input_ids'].shape}")
        print(f"  Text attention_mask: {sample['text_inputs']['attention_mask'].shape}")
        print(f"  Emotion: {sample['emotion_name']} (label: {sample['emotion_label']})")
        print(f"  Sentiment: {sample['sentiment_name']} (label: {sample['sentiment_label']})")
        print(f"  Raw text: '{sample['raw_text'][:50]}...'")
    
    # Test data loader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_v1)
    for batch in loader:
        if batch is not None:
            print("\nBatch shapes:")
            print(f"  Video: {batch['video_frames'].shape}")
            print(f"  Audio: {batch['audio_features'].shape}")
            print(f"  Text: {batch['text_inputs']['input_ids'].shape}")
            print(f"  Metadata available: {list(batch['metadata'].keys())}")
            break
    
    # Display analysis summary
    summary = dataset.get_analysis_summary()
    if summary:
        print(f"\nAnalysis Summary:")
        print(f"  Dataset Version: {summary['dataset_version']}")
        print(f"  Total Samples: {summary['total_samples']}")
        print(f"  Emotion Classes: {summary['emotion_classes']}")
        print(f"  Sentiment Classes: {summary['sentiment_classes']}")
        print(f"  Analysis Files Created: {len(summary['analysis_files'])}")