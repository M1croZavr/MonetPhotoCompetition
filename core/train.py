import torch
from torch import nn
from torchvision.utils import save_image
from tqdm.auto import tqdm
from models import Discriminator, Generator
from utils.dataset import MonetPhotoDataset
from utils.config import (LEARNING_RATE, DEVICE, PHOTO_DIR, MONET_DIR, TRANSFORMS, BATCH_SIZE, NUM_EPOCHS, SAVE_MODEL,
                          CHECKPOINT_DISC_MONET, CHECKPOINT_DISC_PHOTO, CHECKPOINT_GEN_MONET, CHECKPOINT_GEN_PHOTO,
                          LAMBDA_CYCLE, LAMBDA_IDENTITY)
from utils.save_load import save_current_state


def train(disc_photo, disc_monet, gen_photo, gen_monet, loader, opt_disc, opt_gen, l1, mse):
    for i, (monet_batch, photo_batch) in tqdm(enumerate(loader)):
        monet_batch = monet_batch.to(DEVICE)
        photo_batch = photo_batch.to(DEVICE)
        disc_photo.train()
        disc_monet.train()
        gen_photo.train()
        gen_monet.train()

        # Generate fake photo from monet painting
        fake_photo = gen_photo(monet_batch)
        # Discriminate fake photo
        disc_photo_fake = disc_photo(fake_photo.detach())
        # Discriminate real photo
        disc_photo_real = disc_photo(photo_batch)
        # Calculate fake generated photo from monet painting discriminator loss
        disc_loss_photo_fake = mse(disc_photo_fake, torch.zeros_like(disc_photo_fake.detach()).to(DEVICE))
        # Calculate real photo discriminator loss
        disc_loss_photo_real = mse(disc_photo_real, torch.ones_like(disc_photo_real.detach()).to(DEVICE))
        disc_loss_photo = disc_loss_photo_fake + disc_loss_photo_real

        fake_monet = gen_monet(photo_batch)
        disc_monet_fake = disc_monet(fake_monet.detach())
        disc_monet_real = disc_monet(monet_batch)
        disc_loss_monet_fake = mse(disc_monet_fake, torch.zeros_like(disc_monet_fake.detach()).to(DEVICE))
        disc_loss_monet_real = mse(disc_monet_real, torch.ones_like(disc_monet_real.detach()).to(DEVICE))
        disc_loss_monet = disc_loss_monet_fake + disc_loss_monet_real

        disc_loss = disc_loss_photo + disc_loss_monet
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        disc_photo_fake = disc_photo(fake_photo)
        disc_monet_fake = disc_monet(fake_monet)
        # Adversarial loss
        gen_loss_photo = mse(disc_photo_fake, torch.ones_like(disc_photo_fake).to(DEVICE))
        gen_loss_monet = mse(disc_monet_fake, torch.ones_like(disc_monet_fake).to(DEVICE))
        # Cycle loss, X (F)-> Y (G)-> X and Y (G)-> X (F)-> Y
        cycle_loss_photo = l1(gen_photo(fake_monet), photo_batch)
        cycle_loss_monet = l1(gen_monet(fake_photo), monet_batch)
        # Identity loss, X (G)-> X, Y (F)-> Y
        # identity_loss_photo = l1(gen_photo(photo_batch), photo_batch)
        # identity_loss_monet = l1(gen_monet(monet_batch), monet_batch)
        # Overall generator loss
        g_loss = gen_loss_photo + gen_loss_monet + \
            LAMBDA_CYCLE * cycle_loss_photo + LAMBDA_CYCLE * cycle_loss_monet  # + \
            # LAMBDA_IDENTITY * identity_loss_photo + LAMBDA_IDENTITY * identity_loss_monet
        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        if i % 256 == 0:
            save_image(fake_photo.squeeze(dim=0) * 0.5 + 0.5, f"../generated_images/fake_photo{i}.png")
            save_image(fake_monet.squeeze(dim=0) * 0.5 + 0.5, f"../generated_images/fake_monet{i}.png")


def main():
    d_monet = Discriminator(3, 64).to(DEVICE)
    d_photo = Discriminator(3, 64).to(DEVICE)
    g_monet = Generator(3, 64).to(DEVICE)
    g_photo = Generator(3, 64).to(DEVICE)
    optimizer_d = torch.optim.Adam(list(d_monet.parameters()) + list(d_photo.parameters()),
                                   lr=LEARNING_RATE,
                                   betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(list(g_monet.parameters()) + list(g_photo.parameters()),
                                   lr=LEARNING_RATE,
                                   betas=(0.5, 0.999))

    l1 = nn.L1Loss()  # For cycle consistency and identity loss function
    # l1 = nn.MSELoss()  # In case when mps is used as training device
    mse = nn.MSELoss()  # For adversarial loss, maybe try log loss later

    train_dataset = MonetPhotoDataset(photo_path=PHOTO_DIR,
                                      monet_path=MONET_DIR,
                                      transforms=TRANSFORMS)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    for epoch in tqdm(range(NUM_EPOCHS)):
        train(disc_photo=d_photo,
              disc_monet=d_monet,
              gen_photo=g_photo,
              gen_monet=g_monet,
              opt_disc=optimizer_d,
              opt_gen=optimizer_g,
              loader=train_loader,
              l1=l1,
              mse=mse)
        if SAVE_MODEL:
            save_current_state(d_monet, optimizer_d, CHECKPOINT_DISC_MONET)
            save_current_state(d_photo, optimizer_d, CHECKPOINT_DISC_PHOTO)
            save_current_state(g_monet, optimizer_g, CHECKPOINT_GEN_MONET)
            save_current_state(g_photo, optimizer_g, CHECKPOINT_GEN_PHOTO)


if __name__ == "__main__":
    main()
