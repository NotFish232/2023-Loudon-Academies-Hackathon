import torch as T
from torch import optim, nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from torch.cuda import amp
from models import Generator, Discriminator
from dataset import PirateDataset
from tqdm import tqdm

BATCH_SIZE = 8
NUM_EPOCHS = 50
GEN_LR = 1e-3
DISC_LR = 1e-3
LATENT_DIM = 2048


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    dataset = PirateDataset(
        transforms=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(device, dtype=T.float32)),
            ]
        )
    )
    train_set, test_set = random_split(dataset, [1, 0])
    train_loader = DataLoader(train_set, BATCH_SIZE, drop_last=True)
    #test_loader = DataLoader(test_set, len(test_set))

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen_optim = optim.Adam(generator.parameters(), GEN_LR)
    disc_optim = optim.Adam(discriminator.parameters(), DISC_LR)

    gen_scaler = amp.GradScaler()
    disc_scaler = amp.GradScaler()

    criterion = nn.BCEWithLogitsLoss()

    ones = T.ones((BATCH_SIZE, 1), device=device)
    zeros = T.zeros((BATCH_SIZE, 1), device=device)

    for epoch in range(1, NUM_EPOCHS + 1):
        acc_gen_loss = 0
        acc_disc_loss = 0
        for imgs in tqdm(train_loader, desc=f"Epoch {epoch}"):
            z = T.randn((BATCH_SIZE, LATENT_DIM), device=device)
            # train generator
            gen_optim.zero_grad()
            with amp.autocast():
                fake_imgs = generator(z)
                y_fake_hat = discriminator(fake_imgs)
                gen_loss = criterion(y_fake_hat, ones)
            gen_scaler.scale(gen_loss).backward()
            gen_scaler.step(gen_optim)
            gen_scaler.update()

            # train discriminator
            disc_optim.zero_grad()
            with amp.autocast():
                real_loss = criterion(discriminator(imgs), ones)
                fake_loss = criterion(discriminator(fake_imgs.detach()), zeros)
                disc_loss = (real_loss + fake_loss) / 2
            disc_scaler.scale(disc_loss).backward()
            disc_scaler.step(disc_optim)
            disc_scaler.update()

            acc_gen_loss += gen_loss.detach()
            acc_disc_loss += disc_loss.detach()
        print(f"Generator loss {acc_gen_loss.item():.2f}")
        print(f"Discriminator loss {acc_disc_loss.item():.2f}")
        

    T.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        },
        "trained_model.pt",
    )

    validation_z = T.randn((5, LATENT_DIM), device=device)
    with T.no_grad():
        imgs = 255 * generator(validation_z)
        for i, img in enumerate(imgs):
            save_image(img, f"img{epoch}-{i}.png")


if __name__ == "__main__":
    main()
