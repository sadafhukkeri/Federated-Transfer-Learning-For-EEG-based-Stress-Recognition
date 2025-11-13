import os
import numpy as np

# Path to your preprocessed data
base_dir = "data\sam40_epochs"   # e.g., "./data/sam40_epochs"
output_dir = "normalized_epochs"
os.makedirs(output_dir, exist_ok=True)

# Loop through all subjects
for subject in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    out_subject_path = os.path.join(output_dir, subject)
    os.makedirs(out_subject_path, exist_ok=True)

    # Loop through stress/relaxed folders
    for state in os.listdir(subject_path):
        state_path = os.path.join(subject_path, state)
        if not os.path.isdir(state_path):
            continue

        out_state_path = os.path.join(out_subject_path, state)
        os.makedirs(out_state_path, exist_ok=True)

        # Load each epoch, normalize, and save
        for file in os.listdir(state_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(state_path, file))

                # Normalize each channel (column-wise)
                mean = np.mean(data, axis=1, keepdims=True)
                std = np.std(data, axis=1, keepdims=True)
                std[std == 0] = 1e-6  # avoid division by zero
                normalized = (data - mean) / std

                np.save(os.path.join(out_state_path, file), normalized)

        print(f"Normalized: {subject} - {state}")
