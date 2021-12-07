from torchgan.losses.wasserstein import WassersteinGradientPenalty
from dataset import KujialeDataset 
from os import makedirs
from os.path import join
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import RandomCrop
from model import Denoiser, DiscriminatorVGG128
from config import config
from torchgan.trainer import Trainer 
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss, WassersteinGradientPenalty
from torch.nn import L1Loss
import datetime
import numpy as np

def train():
    device = 'cuda'
    data = KujialeDataset(
        join('data', 'KJL_features'), 
        join('data', 'KJL_reference'),
        transform=RandomCrop(config['patch_size']))
    data_loader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)

    dir_checkpoint = f'checkpoint/{datetime.datetime.now()}'
    makedirs(dir_checkpoint, exist_ok="true")

    d = DiscriminatorVGG128(3, 64).to(device)
    d_optimizer = optim.Adam(d.parameters(), lr=config['d_lr'])
    d_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=config['milestones'])

    g = Denoiser(3, 7, config['num_xfmr'], config['num_gcp']).to(device)
    g_optimizer = optim.Adam(g.parameters(), lr=config['g_lr'])
    g_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=config['milestones'])

    d_loss_fn = WassersteinDiscriminatorLoss()
    d_loss_p_fn = WassersteinGradientPenalty()
    g_loss_fn = WassersteinGeneratorLoss()
    g_l1_loss_fn = L1Loss()

    d_loss_record = []
    g_loss_record = []

    for epoch in range(config['num_epoch']):
        d_loss_sum = 0.0
        g_loss_sum = 0.0

        for i, (noisy, aux, ref) in enumerate(data_loader):
            batch_size = noisy.shape[0]

            noisy = noisy.to(device)
            aux = aux.to(device)
            ref = ref.to(device)

            denoised = g(noisy, aux)

            d_optimizer.zero_grad()
            d_fake = d(denoised.detach())
            d_real = d(ref)
            epsilon = torch.rand([batch_size, 1, 1, 1]).to(device)
            interpolate = epsilon * denoised.detach() + (1.0 - epsilon) * ref
            interpolate.requires_grad = True
            d_interpolate = d(interpolate)
            d_loss = d_loss_fn(d_real, d_fake) + d_loss_p_fn(interpolate, d_interpolate)
            d_loss.backward()
            d_optimizer.step()
            

            g_optimizer.zero_grad()
            d_fake = d(denoised)
            g_loss = g_loss_fn(d_fake) + g_l1_loss_fn(denoised, ref)
            g_loss.backward()
            g_optimizer.step()
            
            d_scheduler.step()
            g_scheduler.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            d_loss_record.append(d_loss_sum / (i + 1))
            g_loss_record.append(g_loss_sum / (i + 1))

            print(f"\r[INFO] epoch {epoch + 1} [{i + 1}/{len(data_loader)}]:  discriminator loss: {d_loss_record[-1]} | generator loss: {g_loss_record[-1]} ", end='')

        torch.save({
            'd_state_dict': d.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'd_scheduler_state_dict': d_scheduler.state_dict(), 
            'd_loss_record': d_loss_record, 
            'g_state_dict': g.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'g_scheduler_state_dict': g_scheduler.state_dict(), 
            'g_loss_record': g_loss_record, 
        }, join(dir_checkpoint, 'checkpoint.pt'))


if __name__ == '__main__':
    train()
