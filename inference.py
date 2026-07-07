import torch
from pathlib import Path
import os

from feature_extractor import FeatureExtractor
from patchcore import PatchCore
from heatmap import create_heatmap

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = FeatureExtractor(device)
patchcore = PatchCore()

images = [
    Path("../datasets/mvtec/bottle/test/good/000.png"),
    Path("../datasets/mvtec/bottle/test/broken_large/000.png"),
]

os.makedirs("outputs/heatmaps", exist_ok=True)

for image in images:

    print("=" * 60)
    print(f"Testing: {image}")

    # Extract feature map
    feature_map = extractor.extract(image)

    # Prediction
    score, confidence, defect = patchcore.predict(feature_map)

    print(f"Score       : {score:.4f}")
    print(f"Confidence  : {confidence:.2f}%")
    print("Prediction  :", "Defective" if defect else "Normal")

    # Save heatmap
    output_path = f"outputs/heatmaps/{image.stem}_heatmap.png"

    create_heatmap(
        image,
        feature_map,
        output_path
    )

    print(f"Heatmap Saved: {output_path}")