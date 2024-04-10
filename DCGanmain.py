import torch
import torch.nn as nn
import torch.optim as optim
from DCGan import Generator, Discriminator


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 100
batch_size = 64
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# Load dataset
# transform = Compose([Resize((64, 64)), ToTensor()])
# train_dataset = SIDDDataset(root_dir='path/to/sidd/dataset', transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

from DataLoaderManager import DataLoaderManager
from torchvision import transforms


# Example usage:
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

data_loader_manager = DataLoaderManager(root_dir='SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Data/', transform=transform)
dataloader, val_dataloader = data_loader_manager.process_dataloaders(batch_size=32, shuffle=True)




from torchvision.utils import save_image

def save_generated_images(generated_images, epoch, output_dir):
    save_image(generated_images, f"{output_dir}/denoised_images_{epoch+1}.png", nrow=4)


import torch.nn as nn

def discriminator_loss(real_output, fake_output):
    real_loss = nn.MSELoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.MSELoss()(fake_output, torch.zeros_like(fake_output))
    total_loss = (real_loss + fake_loss) / 2
    return total_loss

def generator_loss(fake_output):
    return nn.MSELoss()(fake_output, torch.ones_like(fake_output))

# Initialize models
generator = Generator(in_channels=3, out_channels=3).to(device)
discriminator = Discriminator(in_channels=6).to(device)

# Define optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

for epoch in range(num_epochs):
    for i, (clean_images, noisy_images) in enumerate(dataloader):
        clean_images = clean_images.to(device)
        noisy_images = noisy_images.to(device)

        # Train the discriminator
        d_optimizer.zero_grad()
        real_output = discriminator(torch.cat((clean_images, noisy_images), dim=1))
        fake_input = torch.cat((noisy_images, generator(noisy_images)), dim=1)
        fake_output = discriminator(fake_input.detach())
        d_loss = discriminator_loss(real_output, fake_output)
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        g_optimizer.zero_grad()
        fake_input = torch.cat((noisy_images, generator(noisy_images)), dim=1)
        fake_output = discriminator(fake_input)
        g_loss = generator_loss(fake_output)
        g_loss.backward()
        g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")

    with torch.no_grad():
        denoised_images = generator(noisy_images)
        save_generated_images(denoised_images, epoch, "output_dir")