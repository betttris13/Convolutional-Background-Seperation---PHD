import torch
print(torch.cuda.is_available())

import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import pickle
import numpy as np

name="Crab" #"MSH15_52","Crab","simple_mock" or "mock_DM_signal"

obs_data=pickle.load(open('obs_data/'+name+'.pkl', 'rb'))
print(obs_data.keys())
obs = obs_data["observation"]
sig = obs_data["total_nopoisson"]
back = obs - sig


obs = torch.from_numpy(obs).permute(2, 0, 1).type(torch.float)
sig = sig.transpose(2, 0, 1)
back = back.transpose(2, 0, 1)
print(obs.shape)

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image.
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        latent_in = 49

        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=2, stride=2),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(latent_in, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
    

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

        latent_in = 49
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, latent_in), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=2, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=2, stride=2
            ), 
            act_fn(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, x.shape[0], 7, 7)
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
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # Bound outputs then rescale
        )

    def forward(self, x):
        x = self.net(x)
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
        self.dencoder = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.sig = Head(num_input_channels, base_channel_size, 1, act_fn)
        self.back = Head(num_input_channels, base_channel_size, 1, act_fn)

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.dencoder.forward(x)
        sig_out = self.sig.forward(x)
        back_out = self.back.forward(x)
        return sig_out, back_out


encode = Encoder(obs.shape[0],10,10)
out = encode.forward(obs)
print(out.shape)

decode = Decoder(obs.shape[0],10,10)
result = decode.forward(out)
print(result.shape)

head1 = Head(obs.shape[0],10,1)
head2 = Head(obs.shape[0],10,1)
head_out1 = head1.forward(result).detach().numpy()
head_out2 = head2.forward(result).detach().numpy()
print(head_out1.shape)
print(head_out2.shape)

print(np.sum(np.square(head_out1-sig)))
print(np.sum(np.square(head_out2-back)))

full_net = Seperator(obs.shape[0],10,10, (obs.max()-obs.min()-0.5))
sig_out, back_out = full_net.forward(obs)

sig_out, back_out = sig_out.detach().numpy(), back_out.detach().numpy()

print(np.sum(np.square(head_out1-sig)))
print(np.sum(np.square(head_out2-back)))
