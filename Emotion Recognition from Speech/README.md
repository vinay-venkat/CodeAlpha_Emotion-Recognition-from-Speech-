# Emotion Recognition from Speech (SER)

Deep learning and speech signal processing pipeline to classify human emotions (`neutral`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`) from speech audio using **MFCCs (+ Delta & Delta-Delta)** and **CNN / BiLSTM / CNN-BiLSTM** architectures in PyTorch.

---

## Project Structure

```text
Emotion Recognition from Speech/
├── app.py                          # Interactive Gradio web app (mic + audio upload)
├── requirements.txt                # Project Python dependencies
├── notebooks/
│   └── 01_eda_and_baseline.ipynb   # Exploratory data analysis & baseline comparison
├── scripts/
│   └── create_dummy_data.py        # Synthetic audio generator for quick smoke tests
├── src/
│   ├── config.py                   # Audio, training, path, and label mappings
│   ├── datasets.py                 # Parsers for RAVDESS, TESS, EMO-DB + speaker-aware splits
│   ├── features.py                 # Silence trimming, augmentation, MFCCs, leakage-free scaler
│   ├── models.py                   # SER_CNN, SER_BiLSTM, SER_CNN_BiLSTM (with Attention)
│   ├── train.py                    # End-to-end training, baseline comparison & plotting
│   ├── evaluate.py                 # Held-out test evaluation & confusion matrix
│   └── predict.py                  # CLI inference on any .wav/.mp3 file
├── tests/
│   └── test_pipeline.py            # Automated pytest verification suite
└── data/                           # Place datasets under data/raw/
    └── raw/
        ├── ravdess/
        ├── tess/
        └── emodb/
```

---

## Step-by-Step Setup & Execution (Windows PowerShell)

### Step 1: Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*(Using `python -m pip` avoids Windows Application Control blocks on newly generated `.exe` shims in `.venv\Scripts\`. If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, or invoke `.venv\Scripts\python.exe` directly.)*

### Step 2: Run the Smoke Test First (Before Downloading Large Datasets)

Verify the entire codebase, MFCC extractor, and model forward/backward passes:

```powershell
python -m pytest -v
```

Generate a small synthetic dataset and run a fast 5-epoch training smoke test:

```powershell
python scripts/create_dummy_data.py
python -m src.train --raw-dir data/demo --datasets ravdess --model cnn_lstm --epochs 5 --batch-size 16
```

### Step 3: Download Speech Emotion Datasets

Place any combination of the three supported datasets inside `data/raw/`:

1. **RAVDESS** (English, 24 actors, ~1,440 speech clips)
   - Download `Audio_Speech_Actors_01-24.zip` from Zenodo (`https://zenodo.org/record/1188976`).
   - Extract into `data/raw/ravdess/` (e.g., `data/raw/ravdess/Actor_01/*.wav`).
2. **TESS** (Toronto Emotional Speech Set, 2 female speakers, 2,800 clips)
   - Download from Kaggle / UofT (`https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess`).
   - Extract into `data/raw/tess/` (e.g., `data/raw/tess/OAF_angry/*.wav`).
3. **EMO-DB** (Berlin Database of Emotional Speech, German, 10 speakers, 535 clips)
   - Download from Kaggle / TU Berlin (`https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb`).
   - Extract `.wav` files into `data/raw/emodb/` (e.g., `data/raw/emodb/03a01Fa.wav`).

### Step 4: Explore the Data (Optional Notebook)

Launch Jupyter to inspect class balance, audio durations, waveforms, and MFCC spectrograms:

```powershell
python -m notebook notebooks/01_eda_and_baseline.ipynb
```
*(Or open `notebooks/01_eda_and_baseline.ipynb` directly in your editor and select the `.venv` kernel).*

### Step 5: Train & Compare Models

Train the Deep Neural Network (**MLPClassifier** with 3 dense hidden layers, adaptive learning rate, and L2 regularization) or the Ensemble model on all available speech datasets:

```powershell
python -m src.train --model mlp
```

Or train the soft-voting Ensemble combining Deep MLP + Random Forest:

```powershell
python -m src.train --model ensemble
```

If PyTorch is unblocked on your system (see Smart App Control note below), you can also train temporal deep learning models on raw MFCC time sequences:

```powershell
python -m src.train --model cnn_lstm --epochs 60
python -m src.train --model cnn --epochs 60
python -m src.train --model lstm --epochs 60
```

> [!NOTE]
> **Windows 11 Smart App Control**: If your system has Windows 11 Smart App Control enabled, unsigned `.dll` files in PyTorch wheels may trigger `WinError 4551`. The pipeline automatically falls back to Scikit-Learn's Deep Neural Network (`mlp`), which runs natively without any restrictions. To enable PyTorch, set Smart App Control to *Evaluation Mode* under **Windows Security > App & Browser Control > Smart App Control**.

Training outputs saved automatically:
- `checkpoints/best_model.joblib` (or `best_model.pt` for PyTorch)
- `outputs/metrics.json` (naive baseline vs logistic regression baseline vs trained model metrics)
- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`
- `data/processed/train_split.csv`, `val_split.csv`, `test_split.csv`

### Step 6: Evaluate, Predict & Launch the Web App

Evaluate the saved model on the held-out test split (calculates Accuracy, Macro F1, Weighted F1, and produces the confusion matrix):

```powershell
python -m src.evaluate
```

Predict emotion on any single speech `.wav` or `.mp3` file from CLI:

```powershell
python -m src.predict data\RAVDESS\Actor_01\03-01-05-01-01-01-01.wav
```

Launch the interactive **Gradio Web App** (supports live microphone recording and audio file upload with confidence distributions and MFCC spectrograms):

```powershell
python app.py
```

