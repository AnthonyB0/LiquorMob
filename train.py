# train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, 3, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        return self.out(dec2)

# Define Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        loss = nn.functional.mse_loss(x_features, y_features)
        return loss

# Define the Dataset class
class WatermarkDataset(Dataset):
    def __init__(self, watermarked_dir, clean_dir, transform=None):
        self.watermarked_dir = watermarked_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.watermarked_images = sorted(os.listdir(watermarked_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        assert len(self.watermarked_images) == len(self.clean_images), "Mismatch in number of images."

    def __len__(self):
        return len(self.watermarked_images)

    def __getitem__(self, idx):
        watermarked_path = os.path.join(self.watermarked_dir, self.watermarked_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        watermarked_image = Image.open(watermarked_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")
        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            clean_image = self.transform(clean_image)
        return watermarked_image, clean_image

# Define transformations with data augmentation and resizing to 1200x1200
transform = transforms.Compose([
    transforms.Resize((1200, 1200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

def main():
    print("Loading dataset...")
    train_dataset = WatermarkDataset("dirty_resized", "clean_resized", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)  # Increase batch size and num_workers

    print("Initializing model and optimizer...")
    # Initialize model, loss functions, and optimizer
    model = UNet()
    criterion = nn.MSELoss()
    perceptual_loss = PerceptualLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Load existing model if available
    checkpoint_path = "unet_watermark_removal.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Checkpoint does not contain expected keys, starting from scratch")

    # Training loop
    num_epochs = 5
    total_steps = len(train_loader) * num_epochs
    step = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_start_time = time.time()
        for i, (watermarked, clean) in enumerate(train_loader):
            step_start_time = time.time()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(watermarked)
                mse_loss = criterion(outputs, clean)
                p_loss = perceptual_loss(outputs, clean)
                loss = mse_loss + 0.5 * p_loss  # Adjust weight of perceptual loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step += 1

            if i % 10 == 0:
                elapsed_time = step_end_time - epoch_start_time
                estimated_total_time = (total_steps - step) * step_time
                estimated_time_remaining = estimated_total_time / 60  # in minutes
                print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                print(f"Estimated time remaining: {estimated_time_remaining:.2f} minutes")

        # Save the model checkpoint periodically
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }, checkpoint_path)

    print("Training complete.")

if __name__ == '__main__':
    main()
