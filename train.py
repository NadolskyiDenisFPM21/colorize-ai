import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from config import cfg
from dataset import ColorizationDataset
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils import AverageMeter, save_checkpoint, load_checkpoint, lab_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cGAN colorization model")
    parser.add_argument("--data_dir", type=str, default=cfg.DATA_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--image_size", type=int, default=cfg.IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--lambda_l1", type=float, default=cfg.LAMBDA_L1)
    parser.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size (useful for quick tests)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def build_loaders(
    data_dir: str,
    batch_size: int,
    image_size: int = cfg.IMAGE_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
    max_samples: int = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = ColorizationDataset(data_dir, split="train", image_size=image_size, max_samples=max_samples)
    val_ds = ColorizationDataset(data_dir, split="val", image_size=image_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def train_one_epoch(
    gen: UNetGenerator,
    disc: PatchGANDiscriminator,
    loader: DataLoader,
    opt_gen: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
    criterion_gan: nn.BCEWithLogitsLoss,
    criterion_l1: nn.L1Loss,
    lambda_l1: float,
    device: str,
    writer: SummaryWriter,
    epoch: int,
) -> tuple[float, float]:
    gen.train()
    disc.train()

    meter_d = AverageMeter()
    meter_g = AverageMeter()

    for L, ab_real in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        L = L.to(device)
        ab_real = ab_real.to(device)
        bsz = L.size(0)

        # --- Discriminator step ---
        ab_fake = gen(L).detach()
        real_pred = disc(L, ab_real)
        fake_pred = disc(L, ab_fake)
        loss_d_real = criterion_gan(real_pred, torch.ones_like(real_pred))
        loss_d_fake = criterion_gan(fake_pred, torch.zeros_like(fake_pred))
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        opt_disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # --- Generator step ---
        ab_fake = gen(L)
        fake_pred = disc(L, ab_fake)
        loss_g_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
        loss_g_l1 = criterion_l1(ab_fake, ab_real) * lambda_l1
        loss_g = loss_g_gan + loss_g_l1

        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        meter_d.update(loss_d.item(), bsz)
        meter_g.update(loss_g.item(), bsz)

    global_step = epoch * len(loader)
    writer.add_scalar("Loss/discriminator", meter_d.avg, global_step)
    writer.add_scalar("Loss/generator", meter_g.avg, global_step)

    return meter_d.avg, meter_g.avg


@torch.no_grad()
def log_val_images(
    gen: UNetGenerator,
    loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: str,
    n_samples: int = 4,
) -> None:
    gen.eval()
    L_batch, ab_real_batch = next(iter(loader))
    L_batch = L_batch[:n_samples].to(device)
    ab_real_batch = ab_real_batch[:n_samples].to(device)
    ab_fake_batch = gen(L_batch)

    rgb_real_list = []
    rgb_fake_list = []
    for i in range(L_batch.size(0)):
        rgb_real = lab_to_rgb(L_batch[i].cpu(), ab_real_batch[i].cpu())
        rgb_fake = lab_to_rgb(L_batch[i].cpu(), ab_fake_batch[i].cpu())
        rgb_real_list.append(torch.from_numpy(rgb_real).permute(2, 0, 1))
        rgb_fake_list.append(torch.from_numpy(rgb_fake).permute(2, 0, 1))

    grid_real = make_grid(torch.stack(rgb_real_list).float() / 255.0, nrow=n_samples)
    grid_fake = make_grid(torch.stack(rgb_fake_list).float() / 255.0, nrow=n_samples)
    writer.add_image("Validation/real", grid_real, epoch)
    writer.add_image("Validation/fake", grid_fake, epoch)


def main():
    args = parse_args()
    device = cfg.DEVICE
    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=True)

    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, gen, opt_gen, device=device)
        disc_ckpt = args.resume.replace("gen_", "disc_")
        if os.path.isfile(disc_ckpt):
            load_checkpoint(disc_ckpt, disc, opt_disc, device=device)
        print(f"Resumed from epoch {start_epoch}")
    else:
        latest_gen = os.path.join(cfg.CHECKPOINT_DIR, "gen_latest.pth")
        latest_disc = os.path.join(cfg.CHECKPOINT_DIR, "disc_latest.pth")
        if os.path.isfile(latest_gen):
            start_epoch = load_checkpoint(latest_gen, gen, opt_gen, device=device)
            if os.path.isfile(latest_disc):
                load_checkpoint(latest_disc, disc, opt_disc, device=device)
            print(f"Resumed from epoch {start_epoch} (latest checkpoint)")

    train_loader, val_loader = build_loaders(args.data_dir, args.batch_size, args.image_size, args.num_workers, args.max_samples)
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        loss_d, loss_g = train_one_epoch(
            gen, disc, train_loader, opt_gen, opt_disc,
            criterion_gan, criterion_l1, args.lambda_l1, device, writer, epoch,
        )
        print(f"Epoch [{epoch}/{args.epochs}]  D_loss: {loss_d:.4f}  G_loss: {loss_g:.4f}")

        log_val_images(gen, val_loader, writer, epoch, device)

        if epoch % 5 == 0:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": gen.state_dict(), "optimizer_state_dict": opt_gen.state_dict()},
                cfg.CHECKPOINT_DIR, f"gen_epoch_{epoch:04d}.pth",
            )
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": disc.state_dict(), "optimizer_state_dict": opt_disc.state_dict()},
                cfg.CHECKPOINT_DIR, f"disc_epoch_{epoch:04d}.pth",
            )

        save_checkpoint(
            {"epoch": epoch, "model_state_dict": gen.state_dict(), "optimizer_state_dict": opt_gen.state_dict()},
            cfg.CHECKPOINT_DIR, "gen_latest.pth",
        )
        save_checkpoint(
            {"epoch": epoch, "model_state_dict": disc.state_dict(), "optimizer_state_dict": opt_disc.state_dict()},
            cfg.CHECKPOINT_DIR, "disc_latest.pth",
        )

    writer.close()


if __name__ == "__main__":
    main()
