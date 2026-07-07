from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from feature_extractor import FeatureExtractor

# ---------------------------------
# Dataset
# ---------------------------------

DATASET_PATH = Path("../datasets/mvtec/bottle/train/good")

images = sorted(DATASET_PATH.glob("*.png"))

print(f"Found {len(images)} training images")

# ---------------------------------
# Device
# ---------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# ---------------------------------
# Feature Extractor
# ---------------------------------

extractor = FeatureExtractor(device)

memory_bank = []

print("\nExtracting Features...\n")

for image_path in tqdm(images):

    features = extractor.extract(image_path)

    features = features.reshape(2048, -1).T

    memory_bank.append(features)

memory_bank = np.concatenate(memory_bank, axis=0)

print("\nMemory Bank Shape:", memory_bank.shape)

Path("models").mkdir(exist_ok=True)

np.save("models/memory_bank.npy", memory_bank)

print("Memory Bank Saved!")

# ---------------------------------
# Build NN
# ---------------------------------

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(memory_bank)

# ---------------------------------
# Calculate Training Scores
# ---------------------------------

training_scores = []

TOP_K = 10

print("\nCalculating Training Scores...\n")

for image_path in tqdm(images):

    features = extractor.extract(image_path)

    features = features.reshape(2048, -1).T

    distances, _ = nn.kneighbors(features)

    distances = distances.flatten()

    score = np.mean(np.sort(distances)[-TOP_K:])

    training_scores.append(score)

training_scores = np.array(training_scores)

print("\nTraining Score Statistics")

print("----------------------------")

print("Min :", training_scores.min())
print("Mean:", training_scores.mean())
print("Max :", training_scores.max())

threshold = np.percentile(training_scores, 99)

print("\nThreshold:", threshold)

np.save("models/threshold.npy", np.array([threshold]))

print("\nThreshold Saved!")