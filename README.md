# Multimodal Emotion Recognition on MELD

A tri-modal deep learning system that recognises emotion and sentiment in multi-party conversations by fusing **text**, **audio**, and **video** through a gated multimodal unit. Achieves **56.51% weighted F1** on the MELD benchmark, surpassing the text-only and text+audio baselines reported in the original MELD paper.

## The Problem

Emotion in conversation is expressed jointly through *what* is said, *how* it sounds, and *how the speaker looks*. Most emotion recognition systems consume only one of these channels and consequently miss affective cues that humans pick up instinctively. The MELD dataset captures this complexity with 13,000+ aligned text, audio, and video utterances from multi-party dialogues, but two challenges make it brutal in practice: extreme class imbalance (neutral dominates at 47%, fear and disgust sit at ~2.6% each.
## The Approach

A five-layer architecture takes synchronised text, audio, and video segments and produces joint emotion and sentiment predictions. Each modality has its own encoder: **DistilBERT** for text, **ResNet-50** for video frames, and a **2D CNN** over mel-spectrograms for audio, and all three encoders project to a shared 256-dimensional space. A **Gated Multimodal Unit** then learns per-utterance weights that decide how much each modality contributes to the fused representation. Two classification heads share the fused vector to predict emotion (7 classes) and sentiment (3 classes) jointly.

The system was developed across five experimental versions (V1–V5), each isolating a specific question: how much fine-tuning depth is needed, whether **Optuna**-driven hyperparameter optimisation helps, and whether class balance or dataset volume matters more for multimodal performance.

## Architecture

![Five-layer multimodal architecture](docs/architecture.png)

Data → Processing → Modality Encoders → Gated Fusion → Joint Classification.

## Tech Stack

- **Deep Learning:** Python 3.12, PyTorch 2.4
- **Pretrained Models:** DistilBERT (HuggingFace Transformers), ResNet-50 (TorchVision), custom 2D CNN
- **Audio:** TorchAudio (mel-spectrogram extraction)
- **Video:** OpenCV, FFmpeg (frame extraction at 30 fps)
- **Hyperparameter Search:** Optuna (Tree-structured Parzen Estimator)
- **Evaluation:** scikit-learn, TensorBoard
- **Hardware:** Dual NVIDIA A100 80GB (CUDA 12.9), University of Limerick CSIS cluster

## How It Works

1. **Ingestion** - MELD CSV annotations and MP4 clips loaded with shared `Dialogue_ID` and `Utterance_ID` keys
2. **Preprocessing** -Three parallel pipelines: DistilBERT tokenisation (128 tokens), uniform video frame sampling (30 frames per clip, ImageNet-normalised), and mel-spectrogram extraction (64 frequency bins) via FFmpeg
3. **Encoding** - Each modality passed through its encoder and projected to a 256-dim representation
4. **Fusion** - All three vectors concatenated (768-dim), passed through modality-specific gate networks (sigmoid weights), transformed, gated element-wise, and summed into a single 256-dim fused vector
5. **Classification** -  Dual heads output emotion logits (7 classes) and sentiment logits (3 classes) trained jointly with cross-entropy loss
6. **Output** - Predicted emotion, sentiment, per-class confidences, and modality gate weights for interpretability

## Key Findings

**Multimodal fusion beats the published MELD baselines on weighted F1.**

| Model | Modalities | Weighted F1 |
|---|---|---|
| text-CNN (Poria et al., 2019) | text | 55.02% |
| cMKL (Poria et al., 2019) | text + audio | 55.51% |
| **V4 (this work)** | **text + audio + video** | **56.51%** |

**Each modality specialises for different emotions.** Auxiliary single-modality classifiers trained alongside the fused model revealed consistent patterns: **audio leads on anger** (vocal intensity carries that signal), **video leads on neutral** (facial expression disambiguates ambiguous utterances), and **text leads on the rest**. This isn't a quirk — it replicates across multiple versions and matches intuitions from cognitive science on cross-modal affective processing.

**Dataset volume matters more than perfect class balance.** A version that undersampled every class to 268 samples (the size of the rarest class) for "perfect balance" collapsed to 43.79% F1. A version that instead *removed* the two unlearnable classes (fear, disgust) and balanced the remaining five at 683 samples each recovered minority-class macro F1 to 35.45% — far better than the aggressive undersampling, while still substantially under V4's weighted F1. The takeaway: don't throw away data to achieve balance.

**Optuna cuts training cost without sacrificing accuracy.** Tree-structured Parzen Estimator search over learning rate, batch size, layer unfreezing depth, dropout, and label smoothing produced V4, which reached comparable F1 to V2 in **26% fewer epochs** (25 vs 34).

## Setup

**Requirements:** Python 3.12, CUDA-capable GPU (≥8GB VRAM for inference, 40GB+ recommended for training), MELD dataset.


## License

MIT.
