import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LeakyReLU(0.01), 
            nn.Linear(128, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.disc(input)

class Generator(nn.Module):
    def __init__(self, noise_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dim),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.gen(noise)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
noise_dim = 64
image_dim = 784   #MNIST dataset
batch_size = 32
NUM_EPOCHS = 40

disc = Discriminator(image_dim).to(device)
gen = Generator(noise_dim, image_dim).to(device)
noise = torch.randn((batch_size, noise_dim)).to(device)

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


optim_disc = optim.Adam(disc.parameters(), lr=lr)
optim_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

#tensorboard
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        #discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)   
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        optim_disc.step()

        #generator
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] \
                        Loss Disc: {loss_disc:.3f}, loss Gen: {loss_gen:.3f}"
            )
            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image(
                    "Fake", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real", img_grid_real, global_step=step
                )
                step += 1





