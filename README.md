# Dyslexia Detection

A machine learning project for analyzing reading-audio samples to detect disfluency risk using handcrafted audio features and a Random Forest classifier.

## Overview

This project builds a dyslexia-screening style prototype that classifies uploaded speech samples as either higher disfluency risk or fluent speech.
It includes:
- **Data preprocessing** using custom scripts for chunking disfluent audio and selecting fluent samples
- **Model training** using `librosa` feature extraction and a scikit-learn Random Forest pipeline (`train.py`)
- **Web-based inference app** using Flask for audio upload and real-time analysis

## Features

### Core Features
- **Audio-Based Fluency Screening**: Binary classification between disfluent-risk speech and fluent speech
- **Automated Feature Extraction**: MFCC mean, MFCC standard deviation, zero-crossing rate, and RMS energy from uploaded audio
- **Web Interface**: Simple Flask-based upload form for `.wav` and `.flac` files
- **Complete ML Pipeline**: From preprocessing and dataset balancing to training and deployment

### Advanced Features
- **Chunk-Based Disfluent Processing**: Long disfluent `.wav` files are split into 8-second chunks and silent chunks are removed
- **Balanced Training Input**: Fluent samples are randomly selected to match the disfluent chunk count
- **Group-Aware Data Split**: `GroupShuffleSplit` prevents chunks from the same original disfluent file from leaking across train and test sets
- **Probability-Based Prediction**: The app shows risk probability and inference time instead of only a hard label

## Technologies & Tools Used

### Deep Learning & ML
- **scikit-learn**: Random Forest classifier, evaluation metrics, and grouped train-test split
- **joblib**: Model serialization and loading

### Web Development
- **Flask**: Web application framework
- **Jinja2 Templates**: HTML rendering for upload, result, and error pages
- **Werkzeug**: Secure uploaded filename handling

### Data Processing & Visualization
- **librosa**: Audio loading and feature extraction
- **NumPy**: Padding, stacking, and numerical processing
- **SoundFile**: Writing chunked audio files

### Development Tools
- **Python 3.12.10**: Runtime version listed in `runtime.txt`
- **Git**: Version control

**Basic App Interface:**
- Upload reading audio sample
- Run fluency-risk analysis
- Receive label, probability, and inference time

**Result/Error Interface:**
- Color-coded result display for higher-risk vs fluent outcome
- Dedicated error page for missing files, invalid formats, or processing failures

## Project Structure

### Python Scripts - Workflow:

| Script | Function | Description |
|------|----------|-------------|
| **1** | `chunkdisfluent.py` | Splits raw disfluent `.wav` files into 8-second chunks and skips silent chunks |
| **2** | `select_fluent.py` | Randomly copies 1633 fluent `.flac` files into `dataset/fluent_selected` |
| **3** | `train.py` | Extracts features, performs grouped train-test split, trains Random Forest, prints evaluation, saves `model.pkl` |
| **4** | `app.py` | Loads trained model, accepts uploads, extracts features, predicts disfluency risk, renders HTML pages |

### Flask App Structure:

**app.py** - Main Interface:
- Loads `model.pkl` at startup
- Validates `.wav` and `.flac` uploads
- Extracts 42-dimensional audio feature vector
- Returns prediction probability, label, and inference time

**templates/index.html** - Upload Page:
- Upload form for reading sample
- Accepts supported audio formats
- Prototype screening disclaimer

**templates/result.html** - Result Page:
- Displays classification label
- Shows risk probability
- Shows inference time

**templates/error.html** - Error Page:
- Displays upload or processing errors
- Provides navigation back to home page

### File Structure:

```text
Dyslexia Detection/
|-- app.py                         # Flask inference app
|-- train.py                       # Model training pipeline
|-- chunkdisfluent.py              # Disfluent audio chunking script
|-- select_fluent.py               # Fluent sample selection script
|-- model.pkl                      # Trained Random Forest model
|-- requirements.txt               # Python dependencies
|-- runtime.txt                    # Python runtime version
|-- README.md                      # This file
|-- Demo Video.mp4                 # Project demo video
|-- dataset/
|   |-- disfluent/                 # Raw disfluent WAV recordings (107 files)
|   |-- disfluent_chunks/          # 8-second disfluent WAV chunks (1633 files)
|   |-- fluent/                    # Source fluent FLAC recordings
|   `-- fluent_selected/           # Selected fluent FLAC set (1633 files)
|-- templates/
|   |-- index.html                 # Upload page
|   |-- result.html                # Prediction result page
|   `-- error.html                 # Error page
`-- __pycache__/
```

## Installation & Setup

### Prerequisites
- Python 3.10+ recommended
- `pip` for dependency installation

### Step 1: Clone/Navigate to Project
```bash
cd "E:\VS CODE\Medical\Dyslexia Detection"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
1. Place raw dysfluent `.wav` files inside `dataset/disfluent/`
2. Place fluent `.flac` files inside `dataset/fluent/`
3. Run preprocessing scripts to create:
   - `dataset/disfluent_chunks/`
   - `dataset/fluent_selected/`

## Running the Project

### Training Models

1. **Create disfluent chunks**:
   ```bash
   python chunkdisfluent.py
   ```

2. **Select fluent samples**:
   ```bash
   python select_fluent.py
   ```

3. **Train the model**:
   ```bash
   python train.py
   ```

4. **Expected training flow**:
   - Load chunked disfluent audio and selected fluent audio
   - Extract MFCC and spectral features
   - Split data with `GroupShuffleSplit`
   - Train Random Forest classifier
   - Print confusion matrix and classification report
   - Save trained model as `model.pkl`

**Training Configuration**:
- Audio Duration: 8 seconds
- Sample Rate: 22050 Hz
- Random Forest Trees: 300
- Test Split: 20%
- Random State: 42

#### Flask Web App
```bash
python app.py
```

**Web App Features**:
- Audio upload interface
- `.wav` and `.flac` support
- Real-time disfluency-risk prediction
- Probability score display
- Inference time display

## Instructions for Testing

### Manual Testing

1. **Preprocessing Validation**:
   - Run `chunkdisfluent.py`
   - Verify chunk files are created in `dataset/disfluent_chunks/`
   - Run `select_fluent.py`
   - Verify selected files are created in `dataset/fluent_selected/`

2. **Model Training Validation**:
   - Run `train.py`
   - Verify `model.pkl` is created
   - Check printed confusion matrix and classification report

3. **Web Application Testing**:
   - Start Flask app with `python app.py`
   - Upload valid `.wav` and `.flac` files
   - Verify prediction label, probability, and inference time are displayed
   - Check invalid-upload and empty-upload cases

### Test Cases

#### Functional Tests
- **Audio Upload**: Test `.wav` and `.flac` files
- **Prediction Output**: Confirm label changes based on uploaded sample
- **Probability Display**: Verify probability renders with four decimal places
- **Error Handling**: Test unsupported extensions or missing uploads

#### Performance Tests
- **Response Time**: Confirm inference completes within a few seconds on local machine
- **Memory Usage**: Monitor RAM during feature extraction and prediction
- **Repeated Requests**: Upload multiple files back-to-back and verify stable behavior

### Sample Test Audio
Use files from the included dataset:
```text
dataset/disfluent/F_0101_14y8m_1.wav
dataset/disfluent_chunks/F_0101_14y8m_1_chunk0.wav
dataset/fluent_selected/103-1240-0009.flac
```

## Model Performance

### Expected Results
- **Output Type**: Binary classification with probability score
- **Inference Output**: `Disfluency Risk Detected` or `Fluent Speech`
- **Training Result Summary**: Confusion matrix and classification report are printed after training

### Evaluation Metrics
- Classification report from scikit-learn
- Confusion matrix analysis
- Group-aware holdout evaluation using `GroupShuffleSplit`

## Troubleshooting

### Common Issues

**Model File Missing**:
```bash
python train.py
```

**Invalid File Format**:
- Upload only `.wav` or `.flac` files
- Ensure the browser-selected file is a real audio recording

**Dataset Folder Not Found**:
- Confirm these folders exist:
  - `dataset/disfluent`
  - `dataset/disfluent_chunks`
  - `dataset/fluent`
  - `dataset/fluent_selected`

**No Chunks Created**:
- Check that source `.wav` files are present in `dataset/disfluent/`
- Very quiet segments may be skipped because silent chunks are removed

**Librosa/SoundFile Errors**:
- Reinstall dependencies from `requirements.txt`
- Make sure the uploaded or training file is not corrupted

### Error Messages
- **"No file uploaded."**: Submit the form with an audio file attached
- **"No file selected."**: Choose a file before clicking analyze
- **"Invalid file format."**: Use only WAV or FLAC input
- **Audio decoding/feature extraction errors**: Replace corrupted or unsupported audio files

## Documentation

- **Application Code**: `app.py` contains the inference workflow and Flask routes
- **Training Script**: `train.py` contains feature extraction, grouped split, model training, and saving logic
- **Preprocessing Scripts**: `chunkdisfluent.py` and `select_fluent.py` prepare the dataset before training

## Dataset Information

### Dyslexia/Fluency Audio Dataset
- **Disfluent Raw Audio**: 107 `.wav` recordings in `dataset/disfluent/`
- **Disfluent Chunked Audio**: 1633 `.wav` chunks in `dataset/disfluent_chunks/`
- **Fluent Selected Audio**: 1633 `.flac` files in `dataset/fluent_selected/`
- **Classification**: Binary (disfluency risk `1` vs fluent speech `0`)
- **Format**: WAV and FLAC audio files

### Data Preprocessing
- Resample audio to 22050 Hz
- Trim or pad audio to 8 seconds
- Extract 20 MFCC coefficients
- Compute MFCC mean and standard deviation
- Compute zero-crossing rate
- Compute RMS energy
- Remove silent disfluent chunks during chunk generation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is intended for educational and research use. If you use third-party speech datasets, please follow the original dataset licenses and usage restrictions.

## Contact

For questions or issues, please create an issue in the GitHub repository or contact the development team.

---

**Note**: This project is a prototype screening tool and not a medical diagnosis system. Predictions should not be used as a substitute for professional educational, clinical, or diagnostic assessment.
