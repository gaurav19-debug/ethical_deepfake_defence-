import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

class DeepfakeDetector:
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        # Load pretrained ResNet18 model
        self.model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer for binary classification (real vs deepfake)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.to(self.device)
        self.model.eval()

        # Load custom weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        # image is expected to be a PIL Image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            deepfake_prob = probabilities[1].item()  # index 1 for deepfake class
            real_prob = probabilities[0].item()
            if deepfake_prob > real_prob:
                label = 'Deepfake'
                confidence = deepfake_prob
            else:
                label = 'Real'
                confidence = real_prob
        return label, confidence

    def predict_video(self, video_file):
        # video_file is a file-like object
        # Placeholder logic: randomly assign deepfake or real with confidence
        import random
        deepfake_prob = random.uniform(0, 1)
        if deepfake_prob > 0.5:
            label = 'Deepfake'
            confidence = deepfake_prob
        else:
            label = 'Real'
            confidence = 1 - deepfake_prob
        return label, confidence
