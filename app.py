import os
import time
import numpy as np
import librosa
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------
MODEL_PATH = "model.pkl"
MAX_DURATION = 8
SR = 22050
ALLOWED_EXTENSIONS = {"wav", "flac"}
# ----------------------------------------

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    max_length = MAX_DURATION * SR
    if len(y) > max_length:
        y = y[:max_length]
    else:
        y = np.pad(y, (0, max_length - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc_mean, mfcc_std, zcr, rms])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "audio" not in request.files:
            return render_template("error.html", message="No file uploaded.")

        file = request.files["audio"]

        if file.filename == "":
            return render_template("error.html", message="No file selected.")

        if not allowed_file(file.filename):
            return render_template("error.html", message="Invalid file format.")

        filename = secure_filename(file.filename)
        temp_path = os.path.join("temp_" + filename)
        file.save(temp_path)

        try:
            start = time.time()

            features = extract_features(temp_path).reshape(1, -1)
            prediction = model.predict_proba(features)[0][1]

            inference_time = time.time() - start

            result_label = "Disfluency Risk Detected" if prediction > 0.5 else "Fluent Speech"

            os.remove(temp_path)

            return render_template(
                "result.html",
                result=result_label,
                probability=f"{prediction:.4f}",
                inference_time=f"{inference_time:.4f}",
                risk_high=(prediction > 0.5)
            )

        except Exception as e:
            os.remove(temp_path)
            return render_template("error.html", message=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)