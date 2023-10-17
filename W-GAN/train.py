"""
Training of deep convolutional network with wasserstein1 loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

#Hyperparameters in accordance with WGAN Paper
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
DISC_ITERATIONS = 5
WEIGHT_CLIP = 0.01
TENSORBOARD_LEN = 32

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.ImageFolder(root = "celeb_dataset", transform = transforms)
loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)

gen = Generator(CHANNELS_IMG, Z_DIM, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)  # try different lr for disc and gen?
#tensorboard
fixed_noise = torch.randn((TENSORBOARD_LEN, Z_DIM, 1, 1))
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        print("batch index: ", batch_idx)
        real = real.to(device)
        cur_batch_size = real.shape[0]

        #Discriminator
        for _ in range(DISC_ITERATIONS):
            noise = torch.randn((cur_batch_size, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1) #flatten
            disc_fake = disc(fake).reshape(-1)
            wasserstein_loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            disc.zero_grad()
            wasserstein_loss_disc.backward(retain_graph=True) # retrain_graph=True so that we can reuse the gradients during gen loss
            opt_disc.step()

            #clip parameters beteween 0.01 and -0.01
            for parameter in disc.parameters():
                parameter.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Generator
        disc_fake_gen = disc(fake).reshape(-1)
        wasserstein_loss_gen = -torch.mean(disc_fake_gen)  # for minimizing negative of val is equivalent to maximizing
        gen.zero_grad()
        wasserstein_loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0 and batch_idx != 0:
            gen.eval()
            disc.eval()
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss Disc: {wasserstein_loss_disc:.3f}, Loss Gen: {wasserstein_loss_gen:.3f}"
                  )
            with torch.no_grad():
                fake = gen(noise)
                img_grid_real = torchvision.utils.make_grid(
                    real[:TENSORBOARD_LEN], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step+=1
            gen.train()
            disc.train()
