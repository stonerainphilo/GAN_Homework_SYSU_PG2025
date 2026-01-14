import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# Set CUDA environment variables (set before importing torch)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set device - try CUDA first, fall back to CPU if fails
try:
    device = torch.device("cuda")
    # Verify CUDA availability
    torch.cuda.empty_cache()
    _ = torch.zeros(1).to(device)
except Exception as e:
    print(f"CUDA initialization failed: {e}, falling back to CPU")
    device = torch.device("cpu")
print(f"Using device: {device}")

# Hyperparameter configuration
class Config:
    batch_size = 128
    latent_dim = 100
    img_channels = 1
    img_size = 28
    learning_rate = 0.0002
    beta1 = 0.5
    epochs = 50
    sample_interval = 200

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # MNIST single channel, mean and std both 0.5
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=Config.batch_size, shuffle=True
)

# Generator network definition
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.init_size = Config.img_size // 4  # Initial feature map size
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, Config.img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.model(out)
        return img

# Discriminator network definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(Config.img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate the number of features after conv layers
        # For 28x28 input: 28 -> 14 -> 7 -> 3 -> 1
        # We need to account for padding effects
        with torch.no_grad():
            dummy_input = torch.zeros(1, Config.img_channels, Config.img_size, Config.img_size)
            dummy_output = self.model(dummy_input)
            self.feature_size = dummy_output.view(1, -1).shape[1]
        
        self.adv_layer = nn.Sequential(
            nn.Linear(self.feature_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        # Flatten and connect to the discriminator head
        validity = self.adv_layer(out)
        return validity

# Initialize networks
generator = Generator(Config.latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), 
                         lr=Config.learning_rate, betas=(Config.beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), 
                         lr=Config.learning_rate, betas=(Config.beta1, 0.999))

# Training monitoring
G_losses = []
D_losses = []

# Fixed noise for generating samples
fixed_noise = torch.randn(64, Config.latent_dim, device=device)

# Training function
def train_gan():
    for epoch in range(Config.epochs):
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.shape[0]
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Real images
            real_imgs = imgs.to(device)
            
            # ============ Train Discriminator ============
            optimizer_D.zero_grad()
            
            # Loss on real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, real_labels)
            
            # Generate fake images
            z = torch.randn(batch_size, Config.latent_dim, device=device)
            fake_imgs = generator(z)
            
            # Loss on fake images
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # ============ Train Generator ============
            optimizer_G.zero_grad()
            
            # Generator tries to fool the discriminator
            gen_pred = discriminator(fake_imgs)
            g_loss = adversarial_loss(gen_pred, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Record losses
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            
            # Periodically generate samples
            if i % Config.sample_interval == 0:
                print(f"[Epoch {epoch}/{Config.epochs}] "
                      f"[Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] "
                      f"[G loss: {g_loss.item():.4f}]")
                
                with torch.no_grad():
                    fake = generator(fixed_noise)
                    save_generated_images(epoch, i, fake)
        
        # Save model every 10 epochs
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

# Helper function to save generated images
def save_generated_images(epoch, batch, fake_images, nrow=8):
    fake_images = fake_images.cpu().detach()
    fake_images = (fake_images + 1) / 2  # Denormalize
    
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i].squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Epoch {epoch}, Batch {batch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'generated_epoch_{epoch}_batch_{batch}.png')
    plt.close()

# Function to visualize training progress
def plot_training_progress():
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    plt.savefig('training_loss.png')
    plt.show()

# Function to generate new samples
def generate_new_samples(num_samples=64):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, Config.latent_dim, device=device)
        samples = generator(z).cpu()
        samples = (samples + 1) / 2
    
    # Display generated images
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated MNIST Digits', fontsize=16)
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    print("Starting GAN training...")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Batch size: {Config.batch_size}")
    print(f"Latent dimension: {Config.latent_dim}")
    
    # Start training
    train_gan()
    
    # Plot training progress
    plot_training_progress()
    
    # Generate new samples
    generate_new_samples()
    
    print("Training completed!")