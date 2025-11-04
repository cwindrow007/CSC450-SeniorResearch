# LSTM - Based Chart Generation for Rhythm Games

### <b>OVERVIEW </b>
This project explores the use of <b>Long Short-term Memory (LSTM)</b> neural networks to automatically generate rhythm game charts
specifically for, <i> <b>Hatsune Miku: Project DIVA Mega Mix+ </b></i>.
The goal is to model teh relationship between audio features and chart design patterns, thus producing playable note sequences
that mimic the structure and timing of official <b>SEGA</b> Charts.

The project is divided into two complementary parts:
1. <b>Data Extraction & Preparation</b> - converting in-game chart files and audio into structured training data using a custom tool <b><i>[Miku Melt](https://github.com/cwindrow007/MikuMelt) </i></b>
2. <b>Machine Learning Pipeline </b> - training, evaluating, and generating new rhythm charts using a multi-layer LSTM implemented in PyTorch.
> [!CAUTION]
> Audio, charts, textures, etc. remain property of ***SEGA***/***Crypton Future Media (CFM)*** and will not be included in this repository. 
___

### <b>OBJECTIVES </b>
* Develop a recurrent neural network capable of predicting **note placements** and **note types** from extracted musical features.
* Evaluate the model's ability to reproduce or innovate upon existing chart design styles.
* Demonstrate the potential for AI-assisted chart authoring tools.

___

### <b>Background </b>
Rhythm game charts encode a human sense of timing, repetition, and flow. By training an LSTM on hundreds of real examples,
we can attempt to learn the "charting grammar" that connects **beats** and **note events** overtime.

This model uses features such as onset-strength, mel-spectrograms, and tempo to predict when and which notes occur.

___
### <b>Project Structure </b>

```graphql
SeniorResearch-LSTM/
│
├── data/
│   ├── charts/          # Parsed chart JSONs from Miku Melt
│   ├── audio/           # Corresponding audio (WAV/OGG)
│   ├── features/        # Per-song feature CSVs from librosa
│   └── processed/       # Final merged dataset (.npz)
│
├── models/
│   ├── chart_lstm.py            # LSTM model definition
│   ├── lstm_trained_382songs.pth
│   └── checkpoints/
│
├── scripts/
│   ├── extract_features.py      # Audio → feature vectors
│   ├── prepare_dataset.py       # Chart + feature alignment
│   ├── evaluate.py              # Metrics & visualization
│   ├── predict.py               # Generate new charts
│   └── utils.py
│
├── notebooks/
│   ├── feature_analysis.ipynb
│   └── training_logs.ipynb
│
├── results/
│   ├── metrics.csv
│   ├── training_curves.png
│   └── generated_charts/
│
├── train.py                     # Main training script
├── requirements.txt
├── README.md
└── LICENSE
```
---
## Enviroment Setup
> [!IMPORTANT] The model built will need CUDA support and NVIDIA drivers. You must verify your recognition this way: 
```bash
nvidia-smi
```
---
The following things are needed for this project: 

1. Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```
2. Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install librosa numpy pandas scikit-learn matplotlib tqdm
```
3. Cuda Verification
```bash
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
---
## Data Pipelining
### Phase 1 - Chart extraction
Chart extraction must be performed. It is highly encourgaed to use the MikuMelt Code linked to this repo as a CPK extractor is needed to pull in-game files. Other extractors have been found to be depreciated. Charts are located in the `.dsc` An example of how data is pulled is via the following

```json
{
  "song": "Melt",
  "difficulty": "Hard",
  "notes": [
    {"time": 0.85, "type": "TAP"},
    {"time": 1.12, "type": "HOLD"}
  ]
}
```

### Phase 2 - Audio Extraction
Extracting mel-spectrograms, onset strength, and tempo data is done in the following
```python
import librosa
y, sr = librosa.load("data/audio/melt.wav", sr=22050)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
onset = librosa.onset.onset_strength(y=y, sr=sr)
```

### Phase 3 - Dataset Assembly
Aligning note timestaps with featured frames (≈20 fps) and labeling of each timestep is needed. Each note timestep is delegated and merged as:
```bash
data/processed/diva_382.npz
```
---
## Model Architecture
> [!NOTE] This area is subject to change
``` python
class ChartLSTM(nn.Module):
    def __init__(self, input_size=80, hidden_size=512, num_layers=3, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
```
- Input: 80-dimensional audio feature vector per frame
- Output: Probability distribution across note types.
- Loss: Catagorical Cross entropy
- Optimizer : Adam (1e3 -> 1e4 decay)
- Training: 40-60 epocs on RTX 3080 12Gb

---

## Training and Evaluation
Metrics Computed:
- Accuracy
- Mean timing offset
- Note density correlation
- Loss cuves and confusion matrix

## Dataset Summary
- Songs -> 382
- Avg Duration -> 180s
- Timesteps (20fps) -> 1.3 million
- features per step -> 64-80
- Note classes 5-8

## Results and Future work
- Early experiments achieve promising alignment between AI-generated and human-made note densities.
- Future directions:
-- Transformer or hybrid LSTM-CNN Architectures
-- Dfficulty-specific models
-- interactive chart preview tool
--Cross-game generalization

## Author
Chris Windrow
Senior Research Project - Eastern Connecticut State University
Professor: Garrett Dancik
Fall 2025