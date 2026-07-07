import cv2
import numpy as np


def create_heatmap(image_path, feature_map, output_path):

    # Read original image
    image = cv2.imread(str(image_path))

    image = cv2.resize(image, (256, 256))

    # feature_map shape = (2048,8,8)

    heatmap = np.mean(feature_map, axis=0)

    heatmap -= heatmap.min()

    heatmap /= heatmap.max()

    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (256, 256))

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(output_path, overlay)

    return output_path