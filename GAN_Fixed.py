"""
Basic but Stable DCGAN Implementation
Specifically optimized for your training issues
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters - validated stable settings
class Config:
    # Data parameters
    batch_size = 64
    latent_dim = 100
    img_size = 28
    channels = 1
    
    # Training parameters - critical adjustments!
    g_lr = 0.0002    # Learning rate from original paper
    d_lr = 0.0002    # Same learning rate
    beta1 = 0.5
    
    # Training strategy
    epochs = 30
    d_train_steps = 1
    g_train_steps = 1  # Back to 1:1 training
    
    # Stability enhancements
    label_smoothing = 0.1      # Moderate label smoothing
    noise_std = 0.05           # Discriminator input noise
    dropout_rate = 0.3
    
    sample_interval = 100
    save_interval = 5

# Data loading
print("Loading data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2
)

print(f"Training samples: {len(train_dataset)}")

# Generator - DCGAN style
class DCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = Config.img_size // 4
        self.fc = nn.Linear(Config.latent_dim, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, Config.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator - appropriately weakened
class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(Config.dropout_rate))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(Config.channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        
        # Calculate output size
        ds_size = Config.img_size // 16
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Initialize models
print("\nInitializing models...")
generator = DCGAN_Generator().to(device)
discriminator = DCGAN_Discriminator().to(device)

# Loss function - BCE with label smoothing
criterion = nn.BCELoss()

# Optimizers - using Adam, standard choice for DCGAN
optimizer_G = optim.Adam(generator.parameters(), lr=Config.g_lr, betas=(Config.beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=Config.d_lr, betas=(Config.beta1, 0.999))

# Training monitoring
G_losses = []
D_losses = []
real_accuracies = []
fake_accuracies = []

# Fixed noise for generating samples
fixed_noise = torch.randn(64, Config.latent_dim, device=device)

def train_dcgan():
    """Standard DCGAN training loop"""
    print("\nStarting DCGAN training...")
    print("Configuration:")
    print(f"  Batch size: {Config.batch_size}")
    print(f"  Learning rate: {Config.g_lr}")
    print(f"  Label smoothing: {Config.label_smoothing}")
    print(f"  Dropout rate: {Config.dropout_rate}")
    
    for epoch in range(Config.epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        batch_count = 0
        
        for i, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            batch_count += 1
            real_imgs = real_imgs.to(device)
            
            # ===== Prepare labels =====
            # Label smoothing: real labels 0.9, fake labels 0.1
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)
            
            # ===== Train Discriminator =====
            optimizer_D.zero_grad()
            
            # Real image loss
            real_pred = discriminator(real_imgs)
            d_real_loss = criterion(real_pred, real_labels)
            real_acc = (real_pred > 0.5).float().mean().item()
            
            # Generate fake images
            z = torch.randn(batch_size, Config.latent_dim, device=device)
            fake_imgs = generator(z)
            
            # Fake image loss
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_pred, fake_labels)
            fake_acc = (fake_pred < 0.5).float().mean().item()
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # ===== Train Generator =====
            optimizer_G.zero_grad()
            
            # Generator tries to make discriminator think fake images are real
            gen_pred = discriminator(fake_imgs)
            g_loss = criterion(gen_pred, real_labels)  # Generator wants discriminator output close to 0.9
            
            g_loss.backward()
            optimizer_G.step()
            
            # Record losses and accuracies
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            real_accuracies.append(real_acc)
            fake_accuracies.append(fake_acc)
            
            # Periodic output
            if i % Config.sample_interval == 0:
                avg_g = epoch_g_loss / batch_count
                avg_d = epoch_d_loss / batch_count
                
                print(f"[Epoch {epoch}/{Config.epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D: {avg_d:.4f}] [G: {avg_g:.4f}] "
                      f"[Real Acc: {real_acc:.2%}] [Fake Acc: {fake_acc:.2%}]")
                
                # Generate samples
                with torch.no_grad():
                    generator.eval()
                    samples = generator(fixed_noise).cpu()
                    generator.train()
                    
                    # Save sample images
                    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
                    for idx, ax in enumerate(axes.flat):
                        ax.imshow(samples[idx].squeeze(), cmap='gray')
                        ax.axis('off')
                    plt.suptitle(f'Epoch {epoch}, Batch {i}', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f'dcgan_samples_epoch_{epoch}_batch_{i}.png', dpi=100)
                    plt.close()
        
        # Average losses per epoch
        avg_epoch_g = epoch_g_loss / batch_count
        avg_epoch_d = epoch_d_loss / batch_count
        print(f"Epoch {epoch} completed - Avg D Loss: {avg_epoch_d:.4f}, Avg G Loss: {avg_epoch_g:.4f}")
        
        # Save model checkpoints
        if (epoch + 1) % Config.save_interval == 0:
            torch.save(generator.state_dict(), f'dcgan_generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'dcgan_discriminator_epoch_{epoch}.pth')
            print(f"  Models saved at epoch {epoch}")
    
    return G_losses, D_losses, real_accuracies, fake_accuracies

def analyze_results(G_losses, D_losses, real_acc, fake_acc):
    """Analyze training results"""
    print("\nAnalyzing training results...")
    
    # Create moving average function
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Raw loss curves
    axes[0, 0].plot(G_losses, label='Generator Loss', alpha=0.6, linewidth=1)
    axes[0, 0].plot(D_losses, label='Discriminator Loss', alpha=0.6, linewidth=1)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Raw Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Moving average loss
    window = 100
    if len(G_losses) > window:
        g_smooth = moving_average(G_losses, window)
        d_smooth = moving_average(D_losses, window)
        axes[0, 1].plot(g_smooth, label='Generator (MA)', linewidth=2)
        axes[0, 1].plot(d_smooth, label='Discriminator (MA)', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss (Moving Average)')
        axes[0, 1].set_title('Smoothed Loss Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy curves
    if len(real_acc) > window:
        real_smooth = moving_average(real_acc, window)
        fake_smooth = moving_average(fake_acc, window)
        axes[0, 2].plot(real_smooth, label='Real Accuracy', alpha=0.7)
        axes[0, 2].plot(fake_smooth, label='Fake Accuracy', alpha=0.7)
        axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Discriminator Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Loss ratio
    if len(G_losses) > window:
        g_ma = moving_average(G_losses, window)
        d_ma = moving_average(D_losses, window)
        loss_ratio = []
        for g, d in zip(g_ma, d_ma):
            if d > 0:
                loss_ratio.append(g/d)
            else:
                loss_ratio.append(0)
        
        axes[1, 0].plot(loss_ratio, color='purple', linewidth=2)
        axes[1, 0].axhline(y=1.0, color='r', linestyle='--', label='Ideal (1.0)')
        axes[1, 0].fill_between(range(len(loss_ratio)), 0.3, 3.0, alpha=0.1, color='green')
        axes[1, 0].set_xlabel('Iteration (MA)')
        axes[1, 0].set_ylabel('G Loss / D Loss')
        axes[1, 0].set_title('Loss Ratio (0.3-3.0 is acceptable)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Loss difference
    if len(G_losses) > window:
        loss_diff = moving_average(np.array(G_losses) - np.array(D_losses), window)
        axes[1, 1].plot(loss_diff, color='brown', linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].fill_between(range(len(loss_diff)), -1.0, 1.0, alpha=0.1, color='blue')
        axes[1, 1].set_xlabel('Iteration (MA)')
        axes[1, 1].set_ylabel('G Loss - D Loss')
        axes[1, 1].set_title('Loss Difference (should be small)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Training status diagnosis
    axes[1, 2].axis('off')
    
    # Diagnosis text
    final_g = np.mean(G_losses[-100:]) if len(G_losses) > 100 else G_losses[-1]
    final_d = np.mean(D_losses[-100:]) if len(D_losses) > 100 else D_losses[-1]
    final_real_acc = np.mean(real_acc[-100:]) if len(real_acc) > 100 else real_acc[-1]
    final_fake_acc = np.mean(fake_acc[-100:]) if len(fake_acc) > 100 else fake_acc[-1]
    
    diagnosis = f"""
    Training Diagnosis:
    {'='*40}
    Final Stats (last 100 iterations):
    • Generator Loss: {final_g:.4f}
    • Discriminator Loss: {final_d:.4f}
    • Real Accuracy: {final_real_acc:.2%}
    • Fake Accuracy: {final_fake_acc:.2%}
    • Loss Ratio: {final_g/final_d:.2f} (G/D)
    
    Status Assessment:
    """
    
    # Evaluate training status
    if 0.3 < final_g/final_d < 3.0:
        diagnosis += "✅ Loss ratio is GOOD (0.3-3.0)\n"
    else:
        diagnosis += "⚠️  Loss ratio needs adjustment\n"
    
    if 0.6 < final_real_acc < 0.9 and 0.6 < final_fake_acc < 0.9:
        diagnosis += "✅ Accuracy is GOOD (balanced)\n"
    elif final_real_acc > 0.9:
        diagnosis += "⚠️  Discriminator too good at real images\n"
    elif final_fake_acc > 0.9:
        diagnosis += "⚠️  Discriminator too good at fake images\n"
    else:
        diagnosis += "⚠️  Discriminator too weak\n"
    
    if 0.5 < final_g < 1.5 and 0.5 < final_d < 1.5:
        diagnosis += "✅ Loss values are in good range\n"
    else:
        diagnosis += "⚠️  Loss values need adjustment\n"
    
    axes[1, 2].text(0.1, 0.5, diagnosis, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('dcgan_training_analysis.png', dpi=120, bbox_inches='tight')
    plt.show()

# Generate final samples function
def generate_final_samples(generator, num_samples=64, save_path='final_generated.png'):
    """Generate and display final samples"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, Config.latent_dim, device=device)
        samples = generator(z).cpu()
    
    # Denormalize
    samples = (samples + 1) / 2
    
    # Display
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(samples[idx].squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Final Generated MNIST Digits', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    
    print(f"Final samples saved to {save_path}")
    return samples

# Main program
if __name__ == "__main__":
    # Training
    G_losses, D_losses, real_acc, fake_acc = train_dcgan()
    
    # Analyze results
    analyze_results(G_losses, D_losses, real_acc, fake_acc)
    
    # Generate final samples
    final_samples = generate_final_samples(generator, save_path='dcgan_final_samples.png')
    
    # Save final models
    torch.save(generator.state_dict(), 'dcgan_generator_final.pth')
    torch.save(discriminator.state_dict(), 'dcgan_discriminator_final.pth')
    
    # Save training statistics
    training_stats = {
        'G_losses': G_losses,
        'D_losses': D_losses,
        'real_accuracies': real_acc,
        'fake_accuracies': fake_acc,
        'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
    }
    torch.save(training_stats, 'dcgan_training_stats.pth')
    
    print("\n" + "="*60)
    print("✅ DCGAN Training Completed Successfully!")
    print("="*60)
    print("\nOutput files:")
    print("  • dcgan_training_analysis.png - Training analysis plots")
    print("  • dcgan_final_samples.png - Final generated digits")
    print("  • dcgan_generator_final.pth - Trained generator")
    print("  • dcgan_discriminator_final.pth - Trained discriminator")
    print("  • dcgan_training_stats.pth - Training statistics")
    print("\nTo generate more samples:")
    print("""
    # Load the trained generator
    generator = DCGAN_Generator()
    generator.load_state_dict(torch.load('dcgan_generator_final.pth'))
    generator.eval()
    
    # Generate new samples
    with torch.no_grad():
        z = torch.randn(16, 100)
        new_samples = generator(z)
    """)