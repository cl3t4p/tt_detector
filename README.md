# Table Tennis Sound Detector

A system for detecting and classifying table tennis ball bounces from audio recordings.
It combines signal processing algorithms for **bounce detection** with a CNN-based deep learning model for **surface** and **spin classification**.

For a full technical description, experiments, and results, see the [report](./report.pdf).

---

## Summary

- **Bounce Detection** — 7 algorithms (spectral flux, energy-based, moving average, etc.) that locate bounce events in raw audio
- **Surface Classification** — 13 classes: table, floor, 10 racket types, and other
- **Spin Classification** — 3 classes: backspin, topspin, none
- Pre-trained model checkpoints are provided in `models/`

---

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) as its package manager.

### 1. Clone the repository

```bash
git clone https://github.com/cl3t4p/tt_detector.git
cd tt_detector
```

### 2. Create the virtual environment and install dependencies

```bash
uv sync
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

> **Note:** Python 3.13+ is required. 

---

## Usage

### Bounce Detection

```bash
python main.py
```

Runs a quick test comparing 4 energy calculation methods on a sample audio file.

### Train a Classifier

```bash
python -m src.train_classifier --classify surface --epochs 150
python -m src.train_classifier --classify spin --epochs 150
```

Checkpoints are saved to `models/`. Training is logged via [Weights & Biases](https://wandb.ai).

### Load a Pre-trained Model

```python
from src.classifier.cnn import AudioClassifier

model = AudioClassifier.load_from_checkpoint('models/surface_best.ckpt')
model.eval()
```

### Notebooks

```bash
jupyter notebook notebook/
```

- `01_signal_analysis.ipynb` — Bounce detection exploration
- `02_classification.ipynb` — Classifier training and evaluation
