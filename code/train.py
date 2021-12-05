from dataset import KujialeDataset 
from os.path import join
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop
from model import Denoiser, DiscriminatorVGG128

device = 'cuda'

data = KujialeDataset(
    join('..', 'data', 'KJL_features'), 
    join('..', 'data', 'KJL_reference'),
    transform=RandomCrop(128))


dataloader = DataLoader(data, batch_size=8, shuffle=True)

noisy, aux, ref = next(iter(dataloader))
noisy = noisy.to(device)
aux = aux.to(device)
ref = ref.to(device)

model = Denoiser(3, 7, 5, 2).to(device)
d = DiscriminatorVGG128(3, 64).to(device)

out = model(noisy, aux)
print(out.shape, d(out).shape, d(ref).shape)