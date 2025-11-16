import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm


class CaptchaDataset(Dataset):
    def __init__(self, image_dir, img_size=(64, 200), max_len=6):
        """
        Args:
            image_dir: Directory containing captcha images
            img_size: Target size (height, width)
            max_len: Maximum length of captcha text
        """
        self.image_dir = image_dir
        self.img_size = img_size
        self.max_len = max_len

        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir)
                            if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Build character vocabulary
        self.chars = set()
        for fname in self.image_files:
            # Extract label from filename (e.g., "0a1gfl-0.png" -> "0a1gfl")
            label = os.path.splitext(fname)[0]  # Remove extension
            label = label.rsplit('-', 1)[0]  # Split by last '-' and take first part
            self.chars.update(label.lower())

        self.chars = sorted(list(self.chars))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)

        print(f"Dataset loaded: {len(self.image_files)} images")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(self.chars)}")

        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def text_to_tensor(self, text):
        """Convert text to one-hot encoded tensor"""
        text = text.lower()[:self.max_len]
        # Pad if necessary
        text = text.ljust(self.max_len, ' ')

        # One-hot encoding
        tensor = torch.zeros(self.max_len, self.vocab_size)
        for i, char in enumerate(text):
            if char in self.char_to_idx:
                tensor[i, self.char_to_idx[char]] = 1

        return tensor.flatten()  # Shape: (max_len * vocab_size,)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image and label
        fname = self.image_files[idx]
        # Extract label from filename (e.g., "0a1gfl-0.png" -> "0a1gfl")
        label = os.path.splitext(fname)[0]  # Remove extension
        label = label.rsplit('-', 1)[0]  # Split by last '-' and take first part
        img_path = os.path.join(self.image_dir, fname)

        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Convert label to tensor
        label_tensor = self.text_to_tensor(label)

        return img, label_tensor, label


class Generator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=None, img_size=(64, 200)):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Initial size for upsampling
        self.init_h = img_size[0] // 16  # 4 for 64
        self.init_w = img_size[1] // 16  # 12 for 200

        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256)
        )

        # Combined input processing
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 256, 512 * self.init_h * self.init_w),
            nn.BatchNorm1d(512 * self.init_h * self.init_w),
            nn.ReLU(True)
        )

        # Upsampling blocks
        self.conv_blocks = nn.Sequential(
            # 4x12 -> 8x25
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x25 -> 16x50
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x50 -> 32x100
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x100 -> 64x200
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels
        label_emb = self.label_emb(labels)

        # Concatenate noise and label embedding
        x = torch.cat([noise, label_emb], dim=1)

        # Process through FC layer
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_h, self.init_w)

        # Generate image
        img = self.conv_blocks(x)
        return img


class Discriminator(nn.Module):
    def __init__(self, label_dim=None, img_size=(64, 200)):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 256),
            nn.LeakyReLU(0.2)
        )

        # We'll dynamically create the label image based on actual input size
        # Store img_size for reference but use actual dimensions in forward pass

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # 64x200 -> 32x100
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),  # 4 channels: 3 RGB + 1 label
            nn.LeakyReLU(0.2, inplace=True),

            # 32x100 -> 16x50
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x50 -> 8x25
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x25 -> 4x12
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final classification
        final_h = img_size[0] // 16
        final_w = img_size[1] // 16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * final_h * final_w, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Get actual image dimensions
        batch_size, _, actual_h, actual_w = img.shape

        # Embed labels
        label_emb = self.label_emb(labels)

        # Expand label embedding to image spatial dimensions
        label_img = label_emb.view(batch_size, 256, 1, 1)
        label_img = label_img.expand(batch_size, 256, actual_h, actual_w)

        # Use 1x1 conv to reduce channels from 256 to 1
        if not hasattr(self, 'label_conv'):
            self.label_conv = nn.Conv2d(256, 1, 1).to(img.device)

        label_img = self.label_conv(label_img)

        # Concatenate image and label
        x = torch.cat([img, label_img], dim=1)

        # Process through conv blocks
        x = self.conv_blocks(x)

        # Final classification
        validity = self.fc(x)
        return validity


def train_cgan(data_dir, epochs=200, batch_size=32, lr=0.0002,
               latent_dim=100, img_size=(64, 200), save_interval=10):
    """
    Train conditional GAN on CAPTCHA dataset

    Args:
        data_dir: Directory containing captcha images
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        latent_dim: Dimension of latent noise vector
        img_size: Image size (height, width)
        save_interval: Save model every N epochs
    """

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print(f"Creating dataset from: {data_dir}")
    dataset = CaptchaDataset(data_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    label_dim = dataset.max_len * dataset.vocab_size
    print(f"Label dimension: {label_dim}")

    # Initialize models
    print("Initializing Generator and Discriminator...")
    generator = Generator(latent_dim=latent_dim, label_dim=label_dim, img_size=img_size).to(device)
    discriminator = Discriminator(label_dim=label_dim, img_size=img_size).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr * 0.25, betas=(0.5, 0.999))  # Much slower discriminator

    # Create output directories
    os.makedirs('generated_samples', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("Output directories created: generated_samples/, models/")

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    
    # Training loop
    for epoch in range(epochs):
        epoch_pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                         desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i, (real_imgs, labels, _) in epoch_pbar:
            batch_size = real_imgs.size(0)

            # Move to device
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # Ground truths with stronger label smoothing
            valid = torch.ones(batch_size, 1, device=device) * 0.9  # Soft labels for real
            fake = torch.zeros(batch_size, 1, device=device) + 0.1  # Add noise to fake labels

            optimizer_G.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z, labels)

            # Generator loss
            g_loss = criterion(discriminator(gen_imgs, labels), valid)

            g_loss.backward()
            optimizer_G.step()

            # =====================
            # Train Discriminator
            # =====================
            optimizer_D.zero_grad()

            # Real images loss
            real_loss = criterion(discriminator(real_imgs, labels), valid)

            # Fake images loss
            fake_loss = criterion(discriminator(gen_imgs.detach(), labels), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Update progress bar
            epoch_pbar.set_postfix({
                'D_loss': f"{d_loss.item():.4f}",
                'G_loss': f"{g_loss.item():.4f}"
            })

        # Save samples and models
        if epoch % save_interval == 0:
            print(f"\nSaving samples and model for epoch {epoch}...")
            # Generate sample images
            generator.eval()
            with torch.no_grad():
                # Use actual batch size or minimum of 16
                num_samples = min(16, labels.size(0))
                sample_labels = labels[:num_samples]
                z = torch.randn(num_samples, latent_dim, device=device)
                gen_imgs = generator(z, sample_labels)

                # Save generated images
                from torchvision.utils import save_image
                save_image(gen_imgs, f'generated_samples/epoch_{epoch}.png',
                           nrow=4, normalize=True)
                print(f"Sample images saved to generated_samples/epoch_{epoch}.png")
            generator.train()

            # Save models
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
                'vocab': dataset.char_to_idx
            }, f'models/cgan_epoch_{epoch}.pth')
            print(f"Model saved to models/cgan_epoch_{epoch}.pth")

    print("Training complete!")
    return generator, discriminator, dataset


def generate_captcha(generator, dataset, text, device='cuda', num_samples=1):
    """Generate CAPTCHA images for given text"""
    generator.eval()

    with torch.no_grad():
        label_tensor = dataset.text_to_tensor(text).unsqueeze(0).repeat(num_samples, 1).to(device)

        z = torch.randn(num_samples, generator.latent_dim, device=device)
        gen_imgs = generator(z, label_tensor)

    return gen_imgs


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/cs4243/data/train"
    EPOCHS = 200
    BATCH_SIZE = 32
    IMG_SIZE = (64, 200)

    generator, discriminator, dataset = train_cgan(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_text = "abc123"
    generated = generate_captcha(generator, dataset, sample_text, device, num_samples=4)

    from torchvision.utils import save_image

    save_image(generated, 'sample_generation.png', nrow=2, normalize=True)