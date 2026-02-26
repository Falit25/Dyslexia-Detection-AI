import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import joblib

# ---------------- CONFIG ----------------
MAX_DURATION = 8  # seconds
SR = 22050
DISFLUENT_PATH = "dataset/disfluent_chunks"
FLUENT_PATH = "dataset/fluent_selected"
# ----------------------------------------


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    # Trim or pad to fixed length
    max_length = MAX_DURATION * SR
    if len(y) > max_length:
        y = y[:max_length]
    else:
        y = np.pad(y, (0, max_length - len(y)))

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    # Other spectral features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc_mean, mfcc_std, zcr, rms])


X = []
y_labels = []
groups = []

print("Processing disfluent...")
for file in os.listdir(DISFLUENT_PATH):
    if file.endswith(".wav"):
        file_path = os.path.join(DISFLUENT_PATH, file)
        features = extract_features(file_path)
        X.append(features)
        y_labels.append(1)

        # Group by original file name (before _chunk)
        group_id = file.split("_chunk")[0]
        groups.append(group_id)


print("Processing fluent...")
for file in os.listdir(FLUENT_PATH):
    if file.endswith(".flac"):
        file_path = os.path.join(FLUENT_PATH, file)
        features = extract_features(file_path)
        X.append(features)
        y_labels.append(0)

        # Each fluent file treated as separate group
        groups.append(file)


X = np.array(X)
y_labels = np.array(y_labels)
groups = np.array(groups)

print("Splitting dataset using GroupShuffleSplit...")

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_labels, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_labels[train_idx], y_labels[test_idx]

print("Training model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")