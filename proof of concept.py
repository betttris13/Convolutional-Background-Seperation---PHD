import torch
print(torch.cuda.is_available())

import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os

import data_gen

name="Crab" #"MSH15_52","Crab","simple_mock" or "mock_DM_signal"

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)  
print(device)


# obs_data=pickle.load(open('obs_data/'+name+'.pkl', 'rb'))
# # print(obs_data.keys())
# obs = obs_data["observation"]
# sig = obs_data["total_nopoisson"]
# back = obs - sig


# obs = torch.from_numpy(obs).permute(2, 0, 1).type(torch.float)
# sig = torch.from_numpy(sig).permute(2, 0, 1).type(torch.float)
# back = torch.from_numpy(back).permute(2, 0, 1).type(torch.float)
# print(obs.shape)

batch = 64
nbatch = 200
# nbatch = 10

obs = []
sig = []
back = []

for i in range(batch*nbatch): #9
    if i % batch == 0:
        print(i/batch)
    obs_t, sig_t, back_t = data_gen.gen_image()
    # print(np.sum(sig_t)/np.sum(back_t))
    obs.append(obs_t)
    sig.append(sig_t)
    back.append(back_t)


# obs, sig, back = data_gen.gen_image()

obs = np.stack([obs[i] for i in range(batch*nbatch)])
sig = np.stack([sig[i] for i in range(batch*nbatch)])
back = np.stack([back[i] for i in range(batch*nbatch)])

print(obs.shape)

obs = torch.from_numpy(obs).type(torch.float)
sig = torch.from_numpy(sig).type(torch.float)
back = torch.from_numpy(back).type(torch.float)

obs = obs.to(device)
sig = sig.to(device)
back = back.to(device)

val_obs = []
val_sig = []
val_back = []

for i in range(batch): #9
    val_obs_t, val_sig_t, val_back_t = data_gen.gen_image()
    # print(np.sum(sig_t)/np.sum(back_t))
    val_obs.append(val_obs_t)
    val_sig.append(val_sig_t)
    val_back.append(val_back_t)


# obs, sig, back = data_gen.gen_image()

val_obs = np.stack([val_obs[i] for i in range(batch)])
val_sig = np.stack([val_sig[i] for i in range(batch)])
val_back = np.stack([val_back[i] for i in range(batch)])

print(obs.shape)

val_obs = torch.from_numpy(val_obs).type(torch.float)
val_sig = torch.from_numpy(val_sig).type(torch.float)
val_back = torch.from_numpy(val_back).type(torch.float)

val_obs = val_obs.to(device)
val_sig = val_sig.to(device)
val_back = val_back.to(device)

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image.
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        self.latent_in = 22500 #980
        self.latent_dim = latent_dim
        # latent_in = (input_height // 8) * (input_width // 8) * (2 * base_channel_size)

        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=7, padding=1, stride=2), #47  kernel_size=3 -> 7
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), 
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=2, stride=2),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=2),
            act_fn(),
            # nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  #78
            # act_fn(),
            # nn.Conv3d(1, c_hid, kernel_size=7, padding=1, stride=2),  # Large kernel
            # act_fn(),
            # nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            # act_fn(),
            # nn.Conv3d(c_hid, 2 * c_hid, kernel_size=3, padding=2, stride=2),
            # act_fn(),
            # nn.Conv3d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            # act_fn(),
            # nn.Conv3d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            # act_fn(),
            nn.Flatten(start_dim=1),  # Image grid to single feature vector
            nn.Linear(self.latent_in, latent_dim),
        )
        # self.flatten = nn.Flatten(start_dim=1)
        # self.Linear = nn.Linear(self.latent_in, self.latent_dim)

    def forward(self, x):
        # print(x.shape)
        x = self.net(x)
        # print(x.shape)
        # print(x.shape)
        # x = self.flatten(x)
        # print(x.shape)
        # x = self.Linear(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()

        latent_in = 22500 #980
        self.c_hid = base_channel_size
        self.n = int(np.sqrt(22500/(2*self.c_hid)))
        self.linear = nn.Sequential(nn.Linear(latent_dim, latent_in), act_fn())
        self.net = nn.Sequential(
            # nn.ConvTranspose2d(
            #     2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            # ),
            # act_fn(),
            # nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            # act_fn(),
            nn.ConvTranspose2d(2 * self.c_hid, self.c_hid, kernel_size=3, output_padding=1, padding=2, stride=2),
            act_fn(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=0),
            act_fn(),
            nn.ConvTranspose2d(
                self.c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=2, stride=2
            ), 
            act_fn(), 
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1 ), # added nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1 )
            act_fn(),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.linear(x)
        # print(self.n)
        x = x.reshape(x.shape[0], 2*self.c_hid, self.n, self.n)
        # x = x.reshape(x.shape[0], 20, 7, 7)
        # x = x.reshape(x.shape[0], -1, 7, 7)
        # print(x.shape)
        x = self.net(x)
        return x
    
class Head(nn.Module):

    def __init__(self, num_input_channels: int, base_channel_size: int, scale: float, act_fn: object = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()

        c_hid = base_channel_size
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1 ),
            act_fn(),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1), # 4
            act_fn(), 
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # Bound outputs then rescale
            # act_fn(), # 51
        )

    def forward(self, x):
        x = self.net(x)
        # x = 0.5*x + 1.0
        x = x * self.scale
        return x


class Seperator(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, scale: float, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image.
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()

        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim, act_fn)
        # self.dencoder = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.sig = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn) #Head(num_input_channels, base_channel_size, scale, act_fn)
        self.back = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn) #Head(num_input_channels, base_channel_size, scale, act_fn)
        # self.sig = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn)
        # self.back = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn)

    def forward(self, x):
        x = self.encoder.forward(x)
        # x = self.dencoder.forward(x)
        back_out = self.back.forward(x)
        sig_out = self.sig.forward(x)
        sum = torch.sum(sig_out, (1,2,3)) #+ 10.0*torch.sum(back_out, (1,2,3))
        # sig_out = sig_out / sum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # back_out = back_out / sum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return sig_out, back_out
        # return x
    
    def _get_reconstruction_loss(self, batch):
        x, sig_in, back_in = batch
        # back_in = x - sig_in
        sig_out, back_out = self.forward(x)
        # y = self.forward(x)
        # loss = F.mse_loss(x.unsqueeze(0), y, reduction="none")
        sig_loss = F.mse_loss(sig_in, sig_out, reduction="none")
        back_loss = F.mse_loss(back_in, back_out, reduction="none")
        # # print(sig_loss.shape)
        # loss = loss.sum().mean()
        sig_loss = sig_loss.sum().mean()
        back_loss = back_loss.sum().mean()
        loss = sig_loss + back_loss
        # loss = back_loss
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        # self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)


# Testing forward pass and loss
print(obs.min(), obs.max())
model = Seperator(obs.shape[1],50,10000, (obs.max())) # 49 Base channel = 10 -> 25
model.to(device)
# sig_out, back_out = model.forward(obs)

# print(np.sum(np.square(sig_out.detach().numpy()-sig.detach().numpy())))
# print(np.sum(np.square(back_out.detach().numpy()-back.detach().numpy())))
# print(np.sum(np.square(sig_out.detach().numpy()-sig.detach().numpy())) + np.sum(np.square(back_out.detach().numpy()-back.detach().numpy())))

# print(model._get_reconstruction_loss((obs,sig)))

lr = 1e-4 #1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
val_losses = []
N = 100
for i in range(N):
    if i%100 == 0:
        print(i/N*100, "%")
    for j in range(nbatch):
        loss = model.training_step((obs[(j*batch):((j+1)*batch),:,:,:], sig[(j*batch):((j+1)*batch),:,:,:], back[(j*batch):((j+1)*batch),:,:,:]), 1)
        # print(loss)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    val_loss = model.validation_step((val_obs, val_sig, val_back), 1)
    val_losses.append(val_loss.item())


sig_out, back_out = model.forward(obs)
obs = obs.cpu().detach().numpy()
sig = sig.cpu().detach().numpy()
back = back.cpu().detach().numpy()
sig_out = sig_out.cpu().detach().numpy()
back_out = back_out.cpu().detach().numpy()

val_sig_out, val_back_out = model.forward(val_obs)
val_obs = val_obs.cpu().detach().numpy()
val_sig = val_sig.cpu().detach().numpy()
val_back = val_back.cpu().detach().numpy()
val_sig_out = val_sig_out.cpu().detach().numpy()
val_back_out = val_back_out.cpu().detach().numpy()


print(obs.shape)

dir = 1

while os.path.isdir(str(dir)):
    print(dir)
    dir += 1

os.makedirs(str(dir))
os.makedirs(str(dir) + "/test")
os.makedirs(str(dir) + "/val")

# print(losses)
plt.plot(losses)
plt.savefig(str(dir) + "/loss.png")
plt.close()

plt.plot(val_losses)
plt.savefig(str(dir) + "/loss_val.png")
plt.close()

for i in range(20):
    plt.imshow(obs[0,i])
    plt.savefig(str(dir) + "/test/obs_"+ str(i) +".png")
    plt.close()
    plt.imshow(sig[0,i])
    plt.savefig(str(dir) + "/test/sig_"+ str(i) +".png")
    plt.close()
    plt.imshow(back[0,i])
    plt.savefig(str(dir) + "/test/back_"+ str(i) +".png")
    plt.close()
    plt.imshow(sig[0,i]+back[0,12])
    plt.savefig(str(dir) + "/test/sig_back_"+ str(i) +".png")
    plt.close()
    plt.imshow(sig_out[0,i])
    plt.savefig(str(dir) + "/test/sig_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(back_out[0,i])
    plt.savefig(str(dir) + "/test/back_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(sig_out[0,i]+back_out[0,i])
    plt.savefig(str(dir) + "/test/sig_out_back_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(obs[0,i]-back_out[0,i])
    plt.savefig(str(dir) + "/test/obs_back_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(obs[0,i]-sig_out[0,i])
    plt.savefig(str(dir) + "/test/obs_sig_out_"+ str(i) +".png")
    plt.close()

for i in range(20):
    plt.imshow(val_obs[0,i])
    plt.savefig(str(dir) + "/val/obs_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_sig[0,i])
    plt.savefig(str(dir) + "/val/sig_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_back[0,i])
    plt.savefig(str(dir) + "/val/back_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_sig[0,i]+val_back[0,12])
    plt.savefig(str(dir) + "/val/sig_back_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_sig_out[0,i])
    plt.savefig(str(dir) + "/val/sig_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_back_out[0,i])
    plt.savefig(str(dir) + "/val/back_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_sig_out[0,i]+val_back_out[0,i])
    plt.savefig(str(dir) + "/val/sig_out_back_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_obs[0,i]-val_back_out[0,i])
    plt.savefig(str(dir) + "/val/obs_back_out_"+ str(i) +".png")
    plt.close()
    plt.imshow(val_obs[0,i]-val_sig_out[0,i])
    plt.savefig(str(dir) + "/val/obs_sig_out_"+ str(i) +".png")
    plt.close()