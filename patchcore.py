import numpy as np
from sklearn.neighbors import NearestNeighbors


class PatchCore:

    def __init__(self):

        self.memory_bank = np.load("models/memory_bank.npy")

        # Keep for display only
        try:
            self.threshold = np.load("models/threshold.npy").item()
        except:
            self.threshold = 24.0

        print("Memory Bank:", self.memory_bank.shape)
        print("Threshold :", self.threshold)

        self.nn = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean"
        )

        self.nn.fit(self.memory_bank)

    def predict(self, features):

        features = features.reshape(2048, -1).T

        distances, _ = self.nn.kneighbors(features)

        distances = distances.flatten()

        TOP_K = 10

        score = np.mean(np.sort(distances)[-TOP_K:])

        # Fixed threshold (same idea as earlier)
        THRESHOLD = 24.0

        is_defect = score > THRESHOLD

        if is_defect:
            confidence = min(70 + ((score - THRESHOLD) / THRESHOLD) * 100, 99.9)
        else:
            confidence = max(99 - ((THRESHOLD - score) / THRESHOLD) * 50, 60)

        return score, confidence, is_defect