# Audio Deepfake Detection

This repository contains implementations of several deep learning models for audio deepfake detection.

The goal is to classify an input audio file as either:

- **Bonafide**: genuine human speech
- **Spoof**: synthetic or manipulated speech

## Models

This repository includes code for the following models:

1. **Log-Mel ResNet18**
2. **Wav2Vec2-based Detector**
3. **AASIST**
4. **XLS-R + SLS**
5. **Score-Level Fusion**

Installation

Install the required packages:

pip install -r requirements.txt

Example requirements.txt:

torch
torchaudio
transformers
librosa
soundfile
numpy
scipy
scikit-learn
pandas
matplotlib
tqdm
PyYAML
Usage
Train Log-Mel ResNet18
python scripts/train_logmel.py --config configs/logmel_resnet18.yaml
Train Wav2Vec2
python scripts/train_wav2vec2.py --config configs/wav2vec2.yaml
Evaluate a Model
python scripts/eval_wav2vec2.py \
  --config configs/wav2vec2.yaml \
  --checkpoint checkpoints/wav2vec2_best.pth \
  --data_dir data/asvspoof2021/LA
Score-Level Fusion
python scripts/score_fusion.py \
  --score_a outputs/wav2vec2_scores.txt \
  --score_b outputs/aasist_scores.txt \
  --alpha 0.2
Results

Main evaluation metric:

EER: Equal Error Rate
AUC: Area Under the ROC Curve

Example results:

Model	ASVspoof2021 LA EER (%)
Log-Mel ResNet18	21.93
Wav2Vec2	4.84
AASIST	4.85
Wav2Vec2 + AASIST Fusion	3.09
Notes
Raw audio datasets are not uploaded.
Large model checkpoints are not uploaded.
Paths in scripts may need to be changed according to your local environment.
This repository is intended for research and educational purposes.
