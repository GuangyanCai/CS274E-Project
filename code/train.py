from dataset import KujialeDataset 
from os.path import join
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import RandomCrop
from model import Denoiser, DiscriminatorVGG128
from config import config
from torchgan.trainer import Trainer 
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss

def train():
    device = 'cuda'
    data = KujialeDataset(
        join('..', 'data', 'KJL_features'), 
        join('..', 'data', 'KJL_reference'),
        transform=RandomCrop(config['patch_size']))
    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)

    d = DiscriminatorVGG128(3, 64).to(device)
    d_optimizer = optim.Adam(d.parameters(), lr=config['d_lr'])
    d_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=config['milestones'])

    g = Denoiser(3, 7, config['num_xfmr'], config['num_gcp']).to(device)
    g_optimizer = optim.Adam(g.parameters(), lr=config['g_lr'])
    g_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=config['milestones'])

    d_loss_fn = WassersteinDiscriminatorLoss()
    g_loss_fn = WassersteinGeneratorLoss()

    for epoch in range(config['num_epoch']):
        for noisy, aux, ref in dataloader:
            noisy = noisy.to(device)
            aux = aux.to(device)
            ref = ref.to(device)

            denoised = g(noisy, aux)

            d_optimizer.zero_grad()
            d_fake = d(denoised.detach())
            d_real = d(ref)
            d_loss = d_loss_fn(d_real, d_fake)
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            d_fake = d(denoised)
            g_loss = g_loss_fn(d_fake)
            g_loss.backward()
            g_optimizer.step()
            
            d_scheduler.step()
            g_scheduler.step()

            print(f"\r[INFO] discriminator loss: {d_loss} | generator loss: {g_loss} ")


if __name__ == '__main__':
    train()