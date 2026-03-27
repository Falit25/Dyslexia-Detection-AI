# Dyslexia Detection Project Report

## 1. Problem Statement

Early signs of dyslexia and related reading disfluency can be difficult to screen consistently through manual observation alone. Traditional assessment often requires time, trained specialists, and repeated testing sessions. This project addresses the need for a simple AI-assisted prototype that analyzes reading audio samples and classifies them as either fluent speech or speech with disfluency risk using audio-based machine learning techniques.

## 2. Objectives

### Primary Objectives:
- Develop a binary classification system for audio-based fluency screening
- Build a complete preprocessing, training, and inference pipeline for speech data
- Create a user-friendly web interface for uploading and analyzing audio files
- Establish a reproducible workflow from raw dataset preparation to saved model deployment

### Secondary Objectives:
- Balance fluent and disfluent training inputs for better learning
- Prevent data leakage by using group-based train-test splitting
- Provide probability-based prediction output instead of only class labels
- Keep the system modular and easy to maintain for future expansion

## 3. Functional Requirements

### 3.1 Major Functional Modules

#### Module 1: Audio Preprocessing Pipeline
**Input:** Raw disfluent `.wav` recordings and fluent `.flac` recordings  
**Output:** Chunked disfluent dataset and selected fluent dataset  
**Functionality:**
- Split long disfluent recordings into 8-second chunks
- Remove silent disfluent chunks
- Randomly sample fluent files to match disfluent chunk count
- Store prepared audio inside project dataset folders

#### Module 2: Feature Extraction and Model Training
**Input:** Prepared disfluent and fluent audio files  
**Output:** Trained `model.pkl` file and evaluation metrics  
**Functionality:**
- Load audio at 22050 Hz
- Trim or pad each sample to fixed 8-second duration
- Extract MFCC, zero-crossing rate, and RMS features
- Train a Random Forest classifier using grouped train-test splitting
- Print confusion matrix and classification report

#### Module 3: Web-Based Inference Interface
**Input:** User-uploaded audio files through the Flask web app  
**Output:** Prediction label, probability score, and inference time  
**Functionality:**
- Accept `.wav` and `.flac` uploads
- Validate file availability and supported formats
- Extract features from uploaded audio
- Load trained model and return disfluency-risk result
- Display result or error through HTML templates

### 3.2 Input/Output Structure

```text
Input Flow:
Raw Audio -> Preprocessing -> Feature Extraction -> Model Inference -> Prediction Result

Training Input:
- Raw disfluent WAV files from dataset/disfluent
- Fluent FLAC files from dataset/fluent
- Chunked disfluent WAV files from dataset/disfluent_chunks
- Selected fluent FLAC files from dataset/fluent_selected

Inference Input:
- Single uploaded audio file
- Supported formats: WAV, FLAC
- Fixed processing window: 8 seconds

Output:
- Binary classification: 1 (Disfluency Risk) or 0 (Fluent Speech)
- Probability score from predict_proba()
- Inference time in seconds
- Error messages for invalid or missing input
```

### 3.3 User Interaction Workflow

1. **Training Workflow:**
   - Prepare disfluent chunks -> Select fluent samples -> Extract features -> Split grouped data -> Train model -> Save `model.pkl`

2. **Inference Workflow:**
   - Open Flask web app -> Upload audio sample -> Run feature extraction -> Generate prediction -> View result page -> Optionally analyze another file

## 4. Non-Functional Requirements

### 4.1 Performance
- **Response Time:** Audio inference should complete within a few seconds on a local machine
- **Throughput:** System should support repeated local testing without restart between uploads
- **Model Processing:** Training should complete on standard CPU hardware without requiring GPU
- **Data Handling:** Fixed-length audio processing should keep runtime predictable

### 4.2 Usability
- **User Interface:** Simple upload-based web interface requiring minimal technical knowledge
- **Accessibility:** Clear result labels and readable error messages
- **Documentation:** Dedicated README and project report for setup and understanding
- **Learning Curve:** Students and evaluators should be able to run the project with basic Python knowledge

### 4.3 Reliability
- **Error Handling:** Graceful handling of missing uploads, empty selections, unsupported formats, and audio-processing failures
- **Prediction Stability:** All files are processed with the same sample rate and duration constraints
- **Validation Logic:** Grouped splitting reduces leakage between related disfluent chunks
- **Model Reusability:** Saved model can be loaded directly for later inference

### 4.4 Scalability
- **Dataset Scalability:** More audio files can be added to the dataset folders with minimal code changes
- **Model Extensibility:** Current pipeline can be replaced with more advanced classifiers later
- **Feature Extensibility:** New acoustic features can be added to the extraction function
- **Deployment Scalability:** Flask app can be adapted for cloud deployment in future versions

### 4.5 Maintainability
- **Code Organization:** Separate scripts for preprocessing, selection, training, and inference
- **Documentation:** Clear folder structure and project-level documentation
- **Modular Design:** Feature extraction logic is reusable across training and inference
- **File Separation:** Templates are separated from backend logic for easier updates

### 4.6 Security
- **Input Validation:** Uploaded files are checked for allowed extensions
- **Filename Safety:** Secure filenames are used before temporary storage
- **Temporary File Removal:** Uploaded audio is deleted after prediction or error handling
- **Scope Control:** The app performs local prediction only and does not expose advanced system operations

### 4.7 Resource Efficiency
- **Fixed-Duration Processing:** Each sample is capped at 8 seconds to control compute cost
- **CPU-Friendly Model:** Random Forest avoids the need for heavy GPU infrastructure
- **Compact Deployment:** Saved `model.pkl` supports lightweight inference
- **Selective Dataset Sampling:** Fluent dataset is reduced to a balanced subset for training efficiency

### 4.8 Error Handling Strategy
- **Upload Validation:** Detects missing file key, missing filename, and unsupported extensions
- **Processing Recovery:** Uses try-except blocks during inference
- **Cleanup Safety:** Temporary uploaded files are removed on both success and failure paths
- **User Feedback:** Dedicated error page displays readable failure messages

## 5. Technical Architecture

### 5.1 System Architecture

```text
+-------------------+     +----------------------+     +----------------------+
|   Web Interface   |<--->|   Inference Engine   |<--->|  Trained RF Model     |
|   (Flask + HTML)  |     | (Feature Extraction) |     |     (model.pkl)       |
+-------------------+     +----------------------+     +----------------------+
         ^                           ^
         |                           |
         |                           v
+-------------------+     +----------------------+     +----------------------+
| User Audio Upload |     | Training Pipeline    |<--->| Prepared Dataset      |
|  WAV / FLAC File  |     | (train.py)           |     | fluent/disfluent sets |
+-------------------+     +----------------------+     +----------------------+
```

### 5.2 Component Architecture

#### Core Components:
1. **Chunking Module:** `chunkdisfluent.py` creates 8-second disfluent chunks and removes silent segments
2. **Selection Module:** `select_fluent.py` balances fluent samples with the disfluent chunk count
3. **Training Engine:** `train.py` extracts features, splits grouped data, trains model, and saves `model.pkl`
4. **Inference Engine:** `app.py` loads model, processes uploads, predicts classes, and renders result pages
5. **UI Layer:** HTML templates provide upload, result, and error screens

### 5.3 Technology Stack

- **Machine Learning Framework:** scikit-learn
- **Audio Processing:** librosa, soundfile
- **Numerical Computing:** NumPy
- **Model Serialization:** joblib
- **Web Framework:** Flask
- **Template Rendering:** Jinja2

## 6. Implementation Details

### 6.1 Model Architecture

#### Random Forest Implementation:
```python
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

### 6.2 Data Preprocessing Pipeline

```python
y, sr = librosa.load(file_path, sr=22050)
max_length = 8 * sr

if len(y) > max_length:
    y = y[:max_length]
else:
    y = np.pad(y, (0, max_length - len(y)))

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcc_mean = np.mean(mfcc.T, axis=0)
mfcc_std = np.std(mfcc.T, axis=0)
zcr = np.mean(librosa.feature.zero_crossing_rate(y))
rms = np.mean(librosa.feature.rms(y=y))
```

### 6.3 Training Configuration

- **Model:** Random Forest Classifier
- **Trees:** 300
- **Split Strategy:** `GroupShuffleSplit`
- **Test Size:** 20%
- **Sample Rate:** 22050 Hz
- **Duration:** 8 seconds per sample
- **Device Requirement:** CPU is sufficient

## 7. Dataset Description

### 7.1 Dataset Overview
- **Raw Disfluent Audio:** 107 `.wav` recordings
- **Chunked Disfluent Audio:** 1633 `.wav` chunks
- **Source Fluent Audio:** 28539 `.flac` recordings
- **Selected Fluent Audio:** 1633 `.flac` files
- **Classification:** Binary (fluent speech vs disfluency risk)

### 7.2 Dataset Structure
```text
dataset/
|-- disfluent/
|   `-- *.wav
|-- disfluent_chunks/
|   `-- *_chunk*.wav
|-- fluent/
|   `-- *.flac
`-- fluent_selected/
    `-- *.flac
```

## 8. Model Selection Rationale

### 8.1 Random Forest
- **Advantages:** Works well on tabular handcrafted features and does not require very large compute resources
- **Audio Suitability:** Suitable for MFCC and spectral summary statistics
- **Interpretation Simplicity:** Easier baseline model to train and deploy than deep neural architectures

### 8.2 Feature-Based Audio Classification Strategy
- **MFCC Features:** Capture spectral speech characteristics related to articulation and fluency patterns
- **Zero-Crossing Rate:** Provides information about signal activity and temporal speech behavior
- **RMS Energy:** Helps capture loudness and speaking-energy variation
- **Grouped Splitting Benefit:** Prevents over-optimistic accuracy caused by chunk-level leakage from the same source file

## 9. Evaluation Methodology

### 9.1 Performance Metrics
- **Primary Metrics:** Confusion matrix and classification report
- **Prediction Score:** Class probability from `predict_proba()`
- **Evaluation Tools:** `classification_report`, `confusion_matrix`

### 9.2 Validation Strategy
- **Split Method:** `GroupShuffleSplit(test_size=0.2, random_state=42)`
- **Grouping Logic:** Disfluent chunks are grouped by original recording name before `_chunk`
- **Reasoning:** Prevents training and test sets from containing related chunks from the same parent recording
- **Model Saving:** Final trained classifier is saved as `model.pkl`

## 10. Results and Performance

### 10.1 Expected Performance
- **Prediction Output:** Two-class result with probability score
- **Inference Speed:** Designed for quick local inference on uploaded audio
- **Training Output:** Confusion matrix and full classification report are printed after each training run

### 10.2 Model Behavior
- The classifier labels audio as either `Disfluency Risk Detected` or `Fluent Speech`
- The web app uses a threshold of `0.5` on the positive-class probability
- Inference time is measured and displayed to the user on the result page

## 11. Deployment and Usage

### 11.1 Installation Requirements
```bash
pip install -r requirements.txt
```

### 11.2 Usage Instructions
1. **Prepare Data:** Run `python chunkdisfluent.py` and `python select_fluent.py`
2. **Train Model:** Execute `python train.py`
3. **Run Web App:** Execute `python app.py`
4. **Inference:** Open the local Flask page in browser and upload a WAV or FLAC file

## 12. Future Enhancements

### 12.1 Technical Improvements
- Replace handcrafted-only features with deep audio embeddings
- Compare Random Forest with SVM, XGBoost, CNN, or transformer-based speech models
- Add cross-validation and richer experiment tracking
- Store evaluation summaries automatically instead of only printing them

### 12.2 Feature Additions
- Audio waveform and spectrogram visualization in the web interface
- Batch upload support for multiple audio files
- Confidence explanation and more detailed feedback
- Cloud deployment and API-based inference

## 13. Commits

The following commit progression reflects the development and documentation history of the project repository:

### 13.1 Initial Development
- **Initial commit: dyslexia fluency analyzer** - `6f23b1a`
- **Add files via upload** - `4ade12e`

### 13.2 Dependency and Environment Updates
- **Add setuptools to requirements.txt** - `9ca3a39`
- **skitilearn upgrade** - `a5bdac6`
- **Fix pkg_resources error by pinning setuptools** - `8912ab2`
- **Fix Render Python version config** - `d716679`

### 13.3 Deployment and Repository Preparation
- **deployment ready** - `507cee5`
- **Ignore demo video** - `29848f6`
- **Add deployment section to README** - `6c4d614`

### 13.4 Documentation Improvements
- **Fix formatting in Project Structure section of README** - `4c27f29`
- **Update README.md** - `3cb58c1`

These commits show the evolution of the project from the initial fluency analyzer implementation to dependency stabilization, deployment preparation, and documentation refinement.

## 14. Conclusion

This project successfully implements a complete machine learning workflow for dyslexia-related reading fluency screening using speech audio. It includes dataset preparation, balanced sampling, grouped model training, and a Flask-based inference interface. The system meets its core objective of providing an accessible prototype that analyzes uploaded audio and returns disfluency-risk predictions in a practical, modular, and extendable form.
