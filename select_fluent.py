import os
import random
import shutil

SOURCE = "dataset/fluent"
TARGET = "dataset/fluent_selected"

os.makedirs(TARGET, exist_ok=True)

files = [f for f in os.listdir(SOURCE) if f.endswith(".flac")]

NUMBER_TO_SELECT = 1633  # Match disfluent chunk count

selected = random.sample(files, NUMBER_TO_SELECT)

for file in selected:
    shutil.copy(os.path.join(SOURCE, file), TARGET)

print("Fluent selection completed.")