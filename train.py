from torchgan.losses.wasserstein import WassersteinGradientPenalty
from dataset import KujialeDataset 
from utils import postprocess_specular, write_tensor_to_exr
from os import makedirs
from os.path import join
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import RandomCrop
from model import Denoiser, DiscriminatorVGG128
from config import config
from torchgan.models import DCGANDiscriminator
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss, WassersteinGradientPenalty
from torch.nn import L1Loss
import datetime
import numpy as np

def train():
    device = 'cuda'
    train_data = KujialeDataset(
        join('data', 'KJL_features'), 
        join('data', 'KJL_reference'),
        transform=RandomCrop(config['patch_size']),
        phase='train')
    train_data_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)

    validate_data = KujialeDataset(
        join('data', 'KJL_features'), 
        join('data', 'KJL_reference'),
        transform=None,
        phase='validate')
    validate_data_loader = DataLoader(validate_data, batch_size=config['validate_batch_size'], shuffle=False)

    curr_time = datetime.datetime.now()
    dir_checkpoint = f'checkpoint/{curr_time}'
    dir_validation = f'validation/{curr_time}'
    makedirs(dir_checkpoint, exist_ok="true")
    makedirs(dir_validation, exist_ok="true")

    # d = DiscriminatorVGG128(3, 64).to(device)
    d = DCGANDiscriminator(in_size=config['patch_size'], in_channels=3).to(device)
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

        fake_acc = 0
        real_acc = 0

        for i, (noisy, aux, ref) in enumerate(train_data_loader):
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
            d_loss = d_loss_fn(d_real, d_fake) + config['d_loss_p_w'] * d_loss_p_fn(interpolate, d_interpolate)
            d_loss.backward()
            d_optimizer.step()

            real_acc += torch.round(d_real).bool().sum().item()
            fake_acc += (1 - torch.round(d_fake)).bool().sum().item()
            
            g_optimizer.zero_grad()
            d_fake = d(denoised)
            g_loss = config['g_loss_w'] * g_loss_fn(d_fake) + config['g_l1_loss_w'] * g_l1_loss_fn(denoised, ref)
            g_loss.backward()
            g_optimizer.step()
            
            d_scheduler.step()
            g_scheduler.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            d_loss_record.append(d_loss_sum / (i + 1))
            g_loss_record.append(g_loss_sum / (i + 1))

            print(f"\r[INFO] epoch {epoch + 1} minibatch [{i + 1}/{len(train_data_loader)}]: discriminator loss: {d_loss_record[-1]} | generator loss: {g_loss_record[-1]}", end='')

        fake_acc /= float(len(train_data))
        real_acc /= float(len(train_data))
        print(f'\n[INFO] discriminator accuracy: {fake_acc} (fake) {real_acc} (real)')

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

        dir_validation_epoch = join(dir_validation, str(epoch))
        makedirs(dir_validation_epoch)

        g.eval()
        with torch.no_grad():
            counter = 0
            for noisy, aux, ref in validate_data_loader:
                if counter >= config['validate_max_num']: break

                noisy = noisy.to(device)
                aux = aux.to(device)

                denoised = g(noisy, aux)

                for n_img, d_img, r_img in zip(noisy, denoised, ref):
                    print(f"\r[INFO] Genereateing validation results [{counter + 1}/{config['validate_max_num']}]", end='')
                    n_img_path = join(dir_validation_epoch, f'{counter}_noisy.exr')
                    d_img_path = join(dir_validation_epoch, f'{counter}_denoised.exr')
                    r_img_path = join(dir_validation_epoch, f'{counter}_reference.exr')
                    write_tensor_to_exr(n_img_path, n_img, postprocess=True)
                    write_tensor_to_exr(d_img_path, d_img, postprocess=True)
                    write_tensor_to_exr(r_img_path, r_img, postprocess=True)
                    counter += 1
            print()
        g.train()


if __name__ == '__main__':
    train()
