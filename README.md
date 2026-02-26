# 🧠 AI-Based Reading Fluency Risk Analyzer

A machine learning web application that analyzes short reading audio samples to detect speech disfluency patterns that may correlate with reading difficulties.

⚠ This is a research prototype and NOT a medical diagnostic tool.

---

## 🚀 Overview

This project implements an end-to-end ML pipeline:

1. Audio preprocessing
2. Feature extraction (MFCC, spectral features)
3. Balanced dataset creation
4. Group-aware train/test splitting
5. Random Forest classification
6. Web-based inference using Flask
7. Cloud deployment ready

The model predicts whether a reading sample shows disfluency patterns.

---

## 🏗 System Architecture

User Upload  
↓  
Flask Backend  
↓  
Audio Feature Extraction  
↓  
Trained Model (`model.pkl`)  
↓  
Prediction  
↓  
Result Display  

Training and deployment are separated:

- Training is done offline.
- The trained model is saved as `model.pkl`.
- The deployed application only performs inference.

---

## 📊 Dataset Strategy

Due to limited availability of labeled dyslexia datasets, the project uses proxy datasets:

- Fluent speech: LibriSpeech subset
- Disfluent speech: UCLASS dataset

Preprocessing steps:

- Long recordings split into 8-second chunks
- Dataset balanced (equal fluent and disfluent samples)
- GroupShuffleSplit used to prevent speaker-level data leakage

This ensures realistic evaluation performance.

---

## 🧪 Model Details

- Algorithm: RandomForestClassifier
- Number of trees: 300
- Feature set:
  - 20 MFCC means
  - 20 MFCC standard deviations
  - Zero Crossing Rate
  - RMS Energy
- Total features per sample: 42
- Group-aware train/test split for fair evaluation

Example evaluation (speaker-held-out):

Accuracy: ~99%  
High recall for disfluency class  

---

## 💻 Tech Stack

- Python 3.10+
- Flask
- Scikit-learn
- Librosa
- NumPy
- Joblib
- Gunicorn (deployment)

---

## 📁 Project Structure
project/
│
├── app.py
├── train.py
├── model.pkl
├── requirements.txt
├── README.md
└── templates/
    ├── index.html
    ├── result.html
    └── error.html
---

## ⚙️ Local Setup

### 1. Clone Repository
```bash
git clone https://github.com/Falit25/Dyslexia-Detection-AI.git
cd project
```
### 2. Create Virtual Environment

Windows:
python3 -m venv env
source env/bin/activate

Mac/Linux:
python3 -m venv env
source env/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt


### 4. Run Application
python app.py

---

## ☁ Deployment (Render Example)

Build Command:pip install -r requirements.txt

Start Command:gunicorn app:app

Note: Free-tier hosting platforms may experience cold-start delays.

---

## 🔍 Limitations

- Uses proxy speech datasets, not clinically labeled dyslexia data.
- Possible dataset-source bias.
- Not validated in real educational settings.
- Not a medical diagnostic system.

---

## 🔮 Future Improvements

- Domain adaptation to reduce dataset bias
- CNN-based spectrogram deep learning model
- Real-time audio streaming support
- Teacher dashboard and progress tracking
- Cross-dataset validation

---

## ⚠ Disclaimer

This tool is intended for research and educational demonstration only. It does not diagnose dyslexia or any medical condition.