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
# Single Character Dataset Class (Class-based folders)
# ============================================================================
class SingleCharCaptchaDataset(Dataset):
    def __init__(self, data_dir, img_size=(64, 64)):
        """
        Creates a dataset from folders organized by class.
        Structure: data_dir/class_name/*.png
        """
        self.data_dir = data_dir
        self.img_size = img_size

        # Get all class folders (0-9, a-z, etc.)
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])

        self.char_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.classes)

        # Build list of (image_path, class_label) tuples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith('.png'):
                    img_path = os.path.join(class_dir, fname)
                    self.samples.append((img_path, class_name))

        print(f"Dataset loaded from class folders")
        print(f"Number of classes: {self.vocab_size}")
        print(f"Classes: {self.classes}")
        print(f"Total samples: {len(self.samples)}")

        # Print samples per class
        class_counts = {}
        for _, class_name in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"Samples per class: {dict(sorted(class_counts.items()))}")

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def char_to_onehot(self, char):
        """Convert character to one-hot encoding."""
        tensor = torch.zeros(self.vocab_size)
        if char in self.char_to_idx:
            tensor[self.char_to_idx[char]] = 1
        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Get label
        label_tensor = self.char_to_onehot(class_name)

        return img, label_tensor, class_name


# ============================================================================
# Single Character Generator
# ============================================================================
class SingleCharGenerator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=36, img_size=(64, 64)):
        super(SingleCharGenerator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Simpler architecture for single characters
        self.init_h = 4
        self.init_w = 4

        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )

        # Combined input processing
        combined_dim = latent_dim + 64
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512 * self.init_h * self.init_w),
            nn.BatchNorm1d(512 * self.init_h * self.init_w),
            nn.ReLU(True)
        )

        # Upsampling layers
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
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

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=1)

        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_h, self.init_w)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        img = self.final(x)

        return img


# ============================================================================
# Single Character Critic
# ============================================================================
class SingleCharCritic(nn.Module):
    def __init__(self, label_dim=36, img_size=(64, 64)):
        super(SingleCharCritic, self).__init__()
        self.img_size = img_size

        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 64),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3)
        )

        # Convolutional path (input is 4 channels: 3 RGB + 1 label)
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LayerNorm([128, 16, 16]),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.3)
        )

        final_h = img_size[0] // 16
        final_w = img_size[1] // 16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * final_h * final_w, 1)
        )

        # Create label projection layer
        self.label_conv = nn.Conv2d(64, 1, 1)

    def forward(self, img, labels):
        batch_size, _, h, w = img.shape

        # Embed and expand labels
        label_emb = self.label_emb(labels)
        label_img = label_emb.view(batch_size, 64, 1, 1).expand(batch_size, 64, h, w)
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
# Training Function
# ============================================================================
def train_single_char_gan(data_dir, epochs=200, batch_size=64,
                          latent_dim=100, img_size=(64, 64), save_interval=10,
                          n_critic=10, lambda_gp=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = SingleCharCaptchaDataset(data_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    label_dim = dataset.vocab_size

    # Initialize models
    generator = SingleCharGenerator(latent_dim=latent_dim, label_dim=label_dim, img_size=img_size).to(device)
    critic = SingleCharCritic(label_dim=label_dim, img_size=img_size).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.00001, betas=(0.5, 0.999))

    os.makedirs('generated_chars', exist_ok=True)
    os.makedirs('models_char', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        for i, (char_imgs, labels, _) in enumerate(dataloader):
            batch_size_actual = char_imgs.size(0)

            char_imgs = char_imgs.to(device)
            labels = labels.to(device)

            # =====================
            # Train Critic
            # =====================
            optimizer_C.zero_grad()

            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_imgs = generator(z, labels).detach()

            real_validity = critic(char_imgs, labels)
            fake_validity = critic(fake_imgs, labels)

            gradient_penalty = compute_gradient_penalty(critic, char_imgs, fake_imgs, labels, device)

            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer_C.step()

            # =================
            # Train Generator
            # =================
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z = torch.randn(batch_size_actual, latent_dim, device=device)
                gen_imgs = generator(z, labels)

                g_loss = -torch.mean(critic(gen_imgs, labels))

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

            # Print progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[C loss: {c_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[Real: {real_validity.mean().item():.4f}] [Fake: {fake_validity.mean().item():.4f}]")

        # Save samples and models
        if epoch % save_interval == 0:
            generator.eval()
            with torch.no_grad():
                # Generate one sample for each character in vocabulary
                num_chars = min(dataset.vocab_size, 36)
                sample_labels = torch.zeros(num_chars, dataset.vocab_size, device=device)
                for j in range(num_chars):
                    sample_labels[j, j] = 1

                z = torch.randn(num_chars, latent_dim, device=device)
                gen_imgs = generator(z, sample_labels)

                from torchvision.utils import save_image
                save_image(gen_imgs, f'generated_chars/epoch_{epoch}.png',
                           nrow=6, normalize=True)
            generator.train()

            torch.save({
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_C': optimizer_C.state_dict(),
                'epoch': epoch,
                'vocab': dataset.char_to_idx
            }, f'models_char/char_gan_epoch_{epoch}.pth')

    print("Training complete!")
    return generator, critic, dataset


# ============================================================================
# Generate Full CAPTCHA from Single Characters
# ============================================================================
def generate_full_captcha(generator, dataset, text, device='cuda', num_samples=1):
    """Generate full captcha by composing individual characters."""
    generator.eval()

    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            chars = []
            for char in text.lower():
                if char in dataset.char_to_idx:
                    label = dataset.char_to_onehot(char).unsqueeze(0).to(device)
                    z = torch.randn(1, generator.latent_dim, device=device)
                    char_img = generator(z, label)
                    chars.append(char_img)

            if chars:
                # Concatenate horizontally
                full_captcha = torch.cat(chars, dim=3)
                results.append(full_captcha)

    if not results:
        return None

    return torch.cat(results, dim=0)


# ============================================================================
# Generate Single Character
# ============================================================================
def generate_character(generator, dataset, char, device='cuda', num_samples=4):
    """Generate multiple samples of a single character."""
    generator.eval()

    if char not in dataset.char_to_idx:
        print(f"Character '{char}' not in vocabulary")
        return None

    with torch.no_grad():
        label = dataset.char_to_onehot(char).unsqueeze(0).repeat(num_samples, 1).to(device)
        z = torch.randn(num_samples, generator.latent_dim, device=device)
        gen_imgs = generator(z, label)

    return gen_imgs


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    DATA_DIR = "/Users/bytedance/Documents/cs4243/labeled_data_train"
    EPOCHS = 200
    BATCH_SIZE = 64
    IMG_SIZE = (64, 64)

    generator, critic, dataset = train_single_char_gan(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        n_critic=10,
        lambda_gp=20
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate individual character samples
    char_samples = generate_character(generator, dataset, 'a', device, num_samples=8)
    if char_samples is not None:
        from torchvision.utils import save_image

        save_image(char_samples, 'sample_char_a.png', nrow=4, normalize=True)

    # Generate full captcha
    sample_text = "abc123"
    full_captcha = generate_full_captcha(generator, dataset, sample_text, device, num_samples=4)
    if full_captcha is not None:
        save_image(full_captcha, 'sample_full_captcha.png', nrow=1, normalize=True)