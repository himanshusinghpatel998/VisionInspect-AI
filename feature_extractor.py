import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class FeatureExtractor:
    def __init__(self, device):
        self.device = device

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove classification layer
        self.model = nn.Sequential(*list(model.children())[:-2])

        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        features = self.model(image)

        return features.squeeze(0).cpu().numpy()