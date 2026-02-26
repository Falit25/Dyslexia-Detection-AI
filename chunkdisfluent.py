import os
import librosa
import soundfile as sf
import numpy as np

INPUT_FOLDER = "dataset/disfluent"
OUTPUT_FOLDER = "dataset/disfluent_chunks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHUNK_DURATION = 8  # seconds

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".wav"):
        file_path = os.path.join(INPUT_FOLDER, file)
        y, sr = librosa.load(file_path, sr=22050)

        chunk_samples = CHUNK_DURATION * sr
        total_chunks = len(y) // chunk_samples

        for i in range(total_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = y[start:end]

            # Remove silent chunks
            if np.mean(np.abs(chunk)) > 0.01:
                output_file = f"{file[:-4]}_chunk{i}.wav"
                sf.write(os.path.join(OUTPUT_FOLDER, output_file), chunk, sr)

print("Disfluent chunking completed.")