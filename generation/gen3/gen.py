import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.autograd as autograd


# ============================================================================
# Dataset Class
# ============================================================================
class CaptchaDataset(Dataset):
    def __init__(self, image_dir, img_size=(64, 200), max_len=6):
        self.image_dir = image_dir
        self.img_size = img_size
        self.max_len = max_len

        self.image_files = [f for f in os.listdir(image_dir)
                            if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Build character vocabulary
        self.chars = set()
        for fname in self.image_files:
            label = os.path.splitext(fname)[0]
            label = label.rsplit('-', 1)[0]
            self.chars.update(label.lower())

        self.chars = sorted(list(self.chars))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)

        print(f"Dataset loaded: {len(self.image_files)} images")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(self.chars)}")

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def text_to_tensor(self, text):
        text = text.lower()[:self.max_len]
        text = text.ljust(self.max_len, ' ')

        tensor = torch.zeros(self.max_len, self.vocab_size)
        for i, char in enumerate(text):
            if char in self.char_to_idx:
                tensor[i, self.char_to_idx[char]] = 1

        return tensor.flatten()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        label = os.path.splitext(fname)[0]
        label = label.rsplit('-', 1)[0]
        img_path = os.path.join(self.image_dir, fname)

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label_tensor = self.text_to_tensor(label)

        return img, label_tensor, label


# ============================================================================
# Improved Generator with Spectral Normalization
# ============================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=216, img_size=(64, 200)):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Calculate initial spatial dimensions
        self.init_h = 4  # Fixed starting point
        self.init_w = 13  # Will give us close to 200 after upsampling

        # Label embedding with batch norm
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128)
        )

        # Combined input processing
        combined_dim = latent_dim + 128
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256 * self.init_h * self.init_w),
            nn.BatchNorm1d(256 * self.init_h * self.init_w),
            nn.ReLU(True)
        )

        # Progressive upsampling
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        # Adaptive pooling to ensure exact output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(img_size)

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=1)

        x = self.fc(x)
        x = x.view(x.size(0), 256, self.init_h, self.init_w)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.final(x)

        # Ensure exact output size
        img = self.adaptive_pool(x)

        return img


# ============================================================================
# Improved Critic (Discriminator) for WGAN-GP
# ============================================================================
class Critic(nn.Module):
    def __init__(self, label_dim=216, img_size=(64, 200)):
        super(Critic, self).__init__()
        self.img_size = img_size

        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        # Main convolutional path
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )

        final_h = img_size[0] // 16
        final_w = img_size[1] // 16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * final_h * final_w, 1)
        )

    def forward(self, img, labels):
        batch_size, _, h, w = img.shape

        # Embed and expand labels
        label_emb = self.label_emb(labels)
        label_img = label_emb.view(batch_size, 128, 1, 1).expand(batch_size, 128, h, w)

        # Reduce label channels
        if not hasattr(self, 'label_conv'):
            self.label_conv = nn.Conv2d(128, 1, 1).to(img.device)
        label_img = self.label_conv(label_img)

        # Concatenate
        x = torch.cat([img, label_img], dim=1)

        # Process through convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        validity = self.fc(x)
        return validity


# ============================================================================
# Gradient Penalty for WGAN-GP
# ============================================================================
def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = critic(interpolates, labels)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ============================================================================
# Training Function with WGAN-GP
# ============================================================================
def train_cgan(data_dir, epochs=200, batch_size=32, lr=0.0001,
               latent_dim=100, img_size=(64, 200), save_interval=10,
               n_critic=5, lambda_gp=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = CaptchaDataset(data_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    label_dim = dataset.max_len * dataset.vocab_size

    # Initialize models
    generator = Generator(latent_dim=latent_dim, label_dim=label_dim, img_size=img_size).to(device)
    critic = Critic(label_dim=label_dim, img_size=img_size).to(device)

    # Optimizers with better settings for stability
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.00005, betas=(0.0, 0.9))

    os.makedirs('generated_samples_1', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        for i, (real_imgs, labels, _) in enumerate(dataloader):
            batch_size_actual = real_imgs.size(0)

            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # =====================
            # Train Critic
            # =====================
            optimizer_C.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_imgs = generator(z, labels).detach()

            # Critic scores
            real_validity = critic(real_imgs, labels)
            fake_validity = critic(fake_imgs, labels)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)

            # Critic loss
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer_C.step()

            # =================
            # Train Generator every n_critic iterations
            # =================
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z = torch.randn(batch_size_actual, latent_dim, device=device)
                gen_imgs = generator(z, labels)

                # Generator loss
                g_loss = -torch.mean(critic(gen_imgs, labels))

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # ADD THIS
                optimizer_G.step()

            # Print progress
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[C loss: {c_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Save samples and models
        if epoch % save_interval == 0:
            generator.eval()
            with torch.no_grad():
                num_samples = min(16, labels.size(0))
                sample_labels = labels[:num_samples]
                z = torch.randn(num_samples, latent_dim, device=device)
                gen_imgs = generator(z, sample_labels)

                from torchvision.utils import save_image
                save_image(gen_imgs, f'generated_samples_1/epoch_{epoch}.png',
                           nrow=4, normalize=True)
            generator.train()

            torch.save({
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_C': optimizer_C.state_dict(),
                'epoch': epoch,
                'vocab': dataset.char_to_idx
            }, f'models/cgan_epoch_{epoch}.pth')

    print("Training complete!")
    return generator, critic, dataset


# ============================================================================
# Generation Function
# ============================================================================
def generate_captcha(generator, dataset, text, device='cuda', num_samples=1):
    generator.eval()

    with torch.no_grad():
        label_tensor = dataset.text_to_tensor(text).unsqueeze(0).repeat(num_samples, 1).to(device)
        z = torch.randn(num_samples, generator.latent_dim, device=device)
        gen_imgs = generator(z, label_tensor)

    return gen_imgs


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    DATA_DIR = "/cs4243/data/train"
    EPOCHS = 200
    BATCH_SIZE = 32
    IMG_SIZE = (64, 200)
    LR = 0.0001

    generator, critic, dataset = train_cgan(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        lr=LR,
        n_critic=5,
        lambda_gp=10
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_text = "abc123"
    generated = generate_captcha(generator, dataset, sample_text, device, num_samples=4)

    from torchvision.utils import save_image

    save_image(generated, 'sample_generation.png', nrow=2, normalize=True)