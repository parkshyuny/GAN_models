import models
import cv2
import os
import torch

from pathlib import Path
from tools import sample_images
from torch import nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(111)

# Initialize generator and discriminator
generator = models.Generator()
discriminator = models.Discriminator()

summary(generator, input_size=(1, 100, 1, 1))
summary(discriminator, input_size=(1, 1, 128, 128))

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

# Configurations
batch_size = 64
image_size = 128

# Configure data loader
class CXRDataset(Dataset): 
    """
    Custom dataset for loading Pneumonia X-ray images.
    """
    def __init__(self, folder: Path):
        self.folder = folder
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.imgs = []

        for img in os.listdir(folder):
            img_path = folder/img
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (image_size, image_size))
                self.imgs.append(img)
            except:
                continue

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]), 1.0 # Dataset contains only pneumonia images, so label is set to 1

folder = Path("./cxr_data_augment/dataset")
ds = CXRDataset(folder/"pneumonia")
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# ----------
#  Training
# ----------

for epoch in range(10):
    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        latent_vecs = torch.normal(0, 1, (batch_size, 100, 1, 1))
        X_generated = generator(latent_vecs)
        y_generated = torch.zeros(batch_size).float()

        X_real = imgs
        y_real = torch.ones(batch_size).float()

        X_all = torch.cat((X_real, X_generated), dim=0)
        y_all = torch.cat((y_real, y_generated), dim=0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        discriminator.zero_grad()

        d_output = discriminator(X_all).view(-1)
        d_loss = adversarial_loss(d_output, y_all)
        d_loss.backward()
        optimizer_d.step()

        # -----------------
        #  Train Generator
        # -----------------
        generator.zero_grad()

        g_output = discriminator(generator(latent_vecs)).view(-1)
        g_loss = adversarial_loss(g_output, y_real)
        g_loss.backward()
        optimizer_g.step()

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")
    latent_vecs = torch.normal(0, 1, (10, 100, 1, 1))
    sample_images(generator(latent_vecs))

