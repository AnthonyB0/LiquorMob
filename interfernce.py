# inference.py

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

# Import UNet architecture from train.py
from train import UNet

# Define WatermarkProcessor class for inference
class WatermarkProcessor:
    def __init__(self, model_path):
        self.model = UNet()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def remove_watermark(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((1200, 1200)),
            transforms.ToTensor(),
        ])
        input_image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_image).squeeze(0)
        output_image = transforms.ToPILImage()(output)
        return output_image

def main():
    test_dir = 'test'
    output_dir = 'output'
    comparison_dir = 'comparison'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    processor = WatermarkProcessor("unet_watermark_removal.pth")

    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(test_dir, filename)
            output_path = os.path.join(output_dir, filename)
            comparison_path = os
