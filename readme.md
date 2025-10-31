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
