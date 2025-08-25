# Model implimentation and helper functions

# from scipy.optimize import curve_fit
# import pandas as pd 
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(torch.cuda.is_available())

import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
# import pickle
# import numpy as np
# from matplotlib import pyplot as plt
# import os

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

# import data_gen

# if torch.cuda.is_available():  
#   dev = "cuda:0" 
# else:  
#   dev = "cpu"  

# device = torch.device(dev)  
# print(device)


# batch = 64
# nbatch = 400
# # nbatch = 10

# obs = []
# sig = []
# back = []
# params = []

# for i in range(batch*nbatch): #9
#     if i % batch == 0:
#         print(i/batch)
#     obs_t, sig_t, back_t, params_t = data_gen.gen_image()
#     # print(np.sum(sig_t)/np.sum(back_t))
#     obs.append(obs_t)
#     sig.append(sig_t)
#     back.append(back_t)
#     params.append(params_t)


# # obs, sig, back = data_gen.gen_image()

# obs = np.stack([obs[i] for i in range(batch*nbatch)])
# sig = np.stack([sig[i] for i in range(batch*nbatch)])
# back = np.stack([back[i] for i in range(batch*nbatch)])
# params = np.stack([params[i] for i in range(batch*nbatch)])

# print(obs.shape)

# obs = torch.from_numpy(obs).type(torch.float)
# sig = torch.from_numpy(sig).type(torch.float)
# back = torch.from_numpy(back).type(torch.float)
# params = torch.from_numpy(params).type(torch.float)


# # obs = obs.to(device)
# # sig = sig.to(device)
# # back = back.to(device)

# val_obs = []
# val_sig = []
# val_back = []
# val_params = []

# for i in range(batch): #9
#     val_obs_t, val_sig_t, val_back_t, val_params_t = data_gen.gen_image()
#     # print(np.sum(sig_t)/np.sum(back_t))
#     val_obs.append(val_obs_t)
#     val_sig.append(val_sig_t)
#     val_back.append(val_back_t)
#     val_params.append(val_params_t)


# # obs, sig, back = data_gen.gen_image()

# val_obs = np.stack([val_obs[i] for i in range(batch)])
# val_sig = np.stack([val_sig[i] for i in range(batch)])
# val_back = np.stack([val_back[i] for i in range(batch)])
# val_params = np.stack([val_params[i] for i in range(batch)])

# print(obs.shape)

# val_obs = torch.from_numpy(val_obs).type(torch.float)
# val_sig = torch.from_numpy(val_sig).type(torch.float)
# val_back = torch.from_numpy(val_back).type(torch.float)

# val_obs = val_obs.to(device)
# val_sig = val_sig.to(device)
# val_back = val_back.to(device)

# ENCODER
class Encoder(nn.Module):
    def __init__(self, in_channels=20, latent_dim=128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1),  # -> [128, 25, 25]
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),

        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [256, 12, 12]
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> [512, 6, 6]
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),

        #     nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # -> [1024, 3, 3]
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        # )

        self.conv = nn.Sequential(
            # Block 1: in_channels -> 128
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> [128, H/2, W/2]

            # Block 2: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> [256, H/4, W/4]

            # Block 3: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> [512, H/8, W/8]

            # Block 4: 512 -> 1024
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> [1024, H/16, W/16]
        )

        self.fc_mu = nn.Linear(1024 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# DECODER
class Decoder(nn.Module):
    def __init__(self, out_channels=20, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 3 * 3)

        # self.deconv = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> [256, 6, 6]
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [128, 12, 12]
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 24, 24]
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=0),  # -> [20, 50, 50]
        #     nn.Sigmoid()  # Output values between 0 and 1
        # )

        self.deconv = nn.Sequential(
            # Block 1: 512 -> 256
            nn.Upsample(scale_factor=2, mode='nearest'),  # [512, H*2, W*2]
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Block 2: 256 -> 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 3: 128 -> 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 4: 64 -> out_channels
            nn.Upsample(size=(50, 50), mode='nearest'),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Output values between 0 and 1
        )

        # self.deconv = nn.Sequential(
        #     # Block 1: 1024 -> 512
        #     nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),  # Upsample
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),

        #     # Block 2: 512 -> 256
        #     nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     # Block 3: 256 -> 128
        #     nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),

        #     # Block 4: 128 -> out_channels
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0),  # Upsample from 24 → 50
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        #     )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 3, 3)
        x = self.deconv(x)
        # x = F.interpolate(x, size=(50, 50), mode='bilinear', align_corners=False)  # Match exact size
        return x

# COMBINED VAE MODEL
class ConvVAE(nn.Module):
    def __init__(self, in_channels=20, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# # model = ConvVAE(in_channels=20, latent_dim=128)
# # x = torch.randn(8, 20, 50, 50)  # batch of 8
# # recon_x, mu, logvar = model(x)
# # loss = model.vae_loss(recon_x, x, mu, logvar)

# # ---------- Hyperparameters ----------
# latent_dim = 512
# batch_size = batch
# epochs = 400
# learning_rate = 1e-5
# alpha = 0.05
# beta = 0.00#1
# debug = False
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ---------- Model ----------
# model = ConvVAE(in_channels=20, latent_dim=latent_dim).to(device)

# # ---------- Optimizer ----------
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # ---------- Loss Function ----------
# def vae_loss(recon_x, x, mu, logvar, alpha = 1.0, beta=1.0):
#     recon_loss = F.mse_loss(recon_x, x, reduction='sum')
#     # penalty_loss = F.l1_loss(recon_x, x, reduction='sum')
#     penalty_loss = torch.norm(recon_x, p=1)
#     kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + beta * kl_div + alpha * penalty_loss

# # ---------- Dummy Dataset (Replace with real one) ----------
# # Input shape: (batch, channels=20, 50, 50)
# # dummy_data = torch.randn(1000, 20, 50, 50)
# # train_loader = DataLoader(TensorDataset(dummy_data), batch_size=batch_size, shuffle=True)

# data_set = TensorDataset(obs, sig, params)  # Midify to alsop have background
# train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

# # ---------- Training Loop ----------
# losses = []
# for epoch in range(1, epochs + 1):
#     model.train()
#     train_loss = 0
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    
#     for input_img, target_img, target_params in progress_bar:
#         input_img = input_img.to(device)
#         target_img = target_img.to(device)
#         optimizer.zero_grad()

#         recon_x, mu, logvar = model(input_img)
#         loss = vae_loss(recon_x, target_img, mu, logvar, alpha, beta)
#         loss.backward()
#         optimizer.step()


#         train_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item() / input_img.size(0))

#     avg_loss = train_loss / len(train_loader.dataset)
#     losses.append(avg_loss)
#     print(f"Epoch {epoch}: Avg Loss = {avg_loss:.2f}")

# data_iter = iter(train_loader)
# obs, sig, params = next(data_iter)
# sig_out, _, _  = model.forward(obs.to(device))
# val_sig_out, _, _  = model.forward(val_obs.to(device))

# obs = obs.numpy()
# sig = sig.numpy()
# back = back.numpy()
# params = params.numpy()
# sig_out= sig_out.cpu().detach().numpy()
# sig_out = sig_out / sig_out.max(axis=(2, 3), keepdims=True)
# # back_out = back_out.cpu().detach().numpy()

# val_obs = val_obs.numpy()
# val_sig = val_sig.numpy()
# val_back = val_back.numpy()
# val_sig_out= val_sig_out.cpu().detach().numpy()
# val_sig_out = val_sig_out / val_sig_out.max(axis=(2, 3), keepdims=True)
# # val_back_out = val_back_out.cpu().detach().numpy()

# dir = 1

# while os.path.isdir(str(dir)):
#     dir += 1

# print("Saving to:", dir)

# os.makedirs(str(dir))
# os.makedirs(str(dir) + "/test")
# os.makedirs(str(dir) + "/val")

# # print(losses)
# plt.plot(losses)
# plt.savefig(str(dir) + "/loss.png")
# plt.close()

# # plt.plot(val_losses)
# # plt.savefig(str(dir) + "/loss_val.png")
# # plt.close()

# if debug == False:
#     print("Saving images...")

#     for i in range(20):
#         plt.imshow(obs[0,i])
#         plt.savefig(str(dir) + "/test/obs_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(sig[0,i])
#         plt.savefig(str(dir) + "/test/sig_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(back[0,i])
#         plt.savefig(str(dir) + "/test/back_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(sig[0,i]+back[0,12])
#         plt.savefig(str(dir) + "/test/sig_back_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(sig_out[0,i])
#         plt.savefig(str(dir) + "/test/sig_out_"+ str(i) +".png")
#         plt.close()
#         # plt.imshow(back_out[0,i])
#         # plt.savefig(str(dir) + "/test/back_out_"+ str(i) +".png")
#         # plt.close()
#         # plt.imshow(sig_out[0,i]+back_out[0,i])
#         # plt.savefig(str(dir) + "/test/sig_out_back_out_"+ str(i) +".png")
#         # plt.close()
#         # plt.imshow(obs[0,i]-back_out[0,i])
#         # plt.savefig(str(dir) + "/test/obs_back_out_"+ str(i) +".png")
#         # plt.close()
#         plt.imshow(obs[0,i]-sig_out[0,i])
#         plt.savefig(str(dir) + "/test/obs_sig_out_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(sig[0,i]-sig_out[0,i])
#         plt.savefig(str(dir) + "/test/sig_sig_out_"+ str(i) +".png")
#         plt.close()

#     for i in range(20):
#         plt.imshow(val_obs[0,i])
#         plt.savefig(str(dir) + "/val/obs_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(val_sig[0,i])
#         plt.savefig(str(dir) + "/val/sig_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(val_back[0,i])
#         plt.savefig(str(dir) + "/val/back_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(val_sig[0,i]+val_back[0,12])
#         plt.savefig(str(dir) + "/val/sig_back_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(val_sig_out[0,i])
#         plt.savefig(str(dir) + "/val/sig_out_"+ str(i) +".png")
#         plt.close()
#         # plt.imshow(val_back_out[0,i])
#         # plt.savefig(str(dir) + "/val/back_out_"+ str(i) +".png")
#         # plt.close()
#         # plt.imshow(val_sig_out[0,i]+val_back_out[0,i])
#         # plt.savefig(str(dir) + "/val/sig_out_back_out_"+ str(i) +".png")
#         # plt.close()
#         # plt.imshow(val_obs[0,i]-val_back_out[0,i])
#         # plt.savefig(str(dir) + "/val/obs_back_out_"+ str(i) +".png")
#         # plt.close()
#         plt.imshow(val_obs[0,i]-val_sig_out[0,i])
#         plt.savefig(str(dir) + "/val/obs_sig_out_"+ str(i) +".png")
#         plt.close()
#         plt.imshow(val_sig[0,i]-val_sig_out[0,i])
#         plt.savefig(str(dir) + "/val/sig_sig_out_"+ str(i) +".png")
#         plt.close()

# def calc_COM(imgs):
#     N, E, X, Y = imgs.shape
#     X_COM_mean = np.zeros(N)
#     Y_COM_mean = np.zeros(N)
    
#     for n in range (N):
#         X_COM = np.zeros(E)
#         Y_COM = np.zeros(E)
#         M = np.zeros(E)
#         for x in range(X):
#             for y in range(Y):
#                 for e in range(E):
#                     m = imgs[n,e,x,y]
#                     X_COM[e] += x * m
#                     Y_COM[e] += y * m
#                     M[e] += m
    
#         for e in range(E):
#             X_COM[e] = X_COM[e]/M[e]
#             Y_COM[e] = Y_COM[e]/M[e]
        
#         X_COM_mean[n] = np.mean(X_COM)
#         Y_COM_mean[n] = np.mean(Y_COM)

#     return X_COM_mean, Y_COM_mean


# print("Calculating COM...")

# X_COM_mean, Y_COM_mean = calc_COM(sig_out)
# sig_X_COM_mean, sig_Y_COM_mean = calc_COM(sig)
# val_X_COM_mean, val_Y_COM_mean = calc_COM(val_sig_out)
# val_sig_X_COM_mean, val_sig_Y_COM_mean = calc_COM(val_sig)

# X_COM = X_COM_mean[0]
# Y_COM = Y_COM_mean[0]
# sig_X_COM = sig_X_COM_mean[0]
# sig_Y_COM = sig_Y_COM_mean[0]
# val_X_COM = val_X_COM_mean[0]
# val_Y_COM = val_Y_COM_mean[0]
# val_sig_X_COM = val_sig_X_COM_mean[0]
# val_sig_Y_COM = val_sig_Y_COM_mean[0]

# print("Calculating brightest...")

# def fit_brightest(img):
#     E, _, _ = img.shape
#     fit = []
#     for i in range(20):
#         fit.append(np.unravel_index(np.argmax(img[i]), img[i].shape))
#     fit = np.nanmean(fit, axis=(0))
#     return fit

# fit = np.zeros((batch, 2))
# val_fit = np.zeros((batch, 2))
# sig_fit = np.zeros((batch, 2))
# sig_val_fit = np.zeros((batch, 2))
# for i in range(batch):
#     temp = fit_brightest(sig_out[i])
#     fit[i, 0] = temp[0]
#     fit[i, 1] = temp[1]
#     temp = fit_brightest(val_sig_out[i])
#     val_fit[i, 0] = temp[0]
#     val_fit[i, 1] = temp[1]
#     temp = fit_brightest(sig[i])
#     sig_fit[i, 0] = temp[0]
#     sig_fit[i, 1] = temp[1]
#     temp = fit_brightest(val_sig[i])
#     sig_val_fit[i, 0] = temp[0]
#     sig_val_fit[i, 1] = temp[1]


# print("Generating full results...")

# results_titles = ["True x", "True y", "Signal brightest x", "Signal brightest y", "Reconstructed brightest x", "Reconstructed brightest y", "Signal COM x", "Signal COM y", "Reconstructed COM x", "Reconstructed COM y", "Error true brightest", "Error true COM", "Error fitted brightest", "Error fitted COM"]
# results = np.zeros((batch, 14))
# val_results = np.zeros((batch, 14))

# for i in range(batch):
#     results[i, 0] = params[i,1]
#     results[i, 1] = params[i,0]
#     results[i, 2] = sig_fit[i,0]
#     results[i, 3] = sig_fit[i,1]
#     results[i, 4] = fit[i,0]
#     results[i, 5] = fit[i,1]
#     results[i, 6] = sig_X_COM_mean[i]
#     results[i, 7] = sig_Y_COM_mean[i]
#     results[i, 8] = X_COM_mean[i]
#     results[i, 9] = Y_COM_mean[i]
#     results[i, 10] = np.sqrt(((params[i,1] - fit[i,0])**2) + ((params[i,0] - fit[i,1])**2))
#     results[i, 11] = np.sqrt(((params[i,1] - X_COM_mean[i])**2) + ((params[i,0] - Y_COM_mean[i])**2))
#     results[i, 12] = np.sqrt((sig_fit[i,0] - fit[i,0])**2) + ((sig_fit[i,1] - fit[i,1])**2)
#     results[i, 13] = np.sqrt((sig_X_COM_mean[i] - X_COM_mean[i])**2) + ((sig_Y_COM_mean[i] - Y_COM_mean[i])**2)

#     val_results[i, 0] = val_params[i,1]
#     val_results[i, 1] = val_params[i,0]
#     val_results[i, 2] = sig_val_fit[i,0]
#     val_results[i, 3] = sig_val_fit[i,1]
#     val_results[i, 4] = val_fit[i,0]
#     val_results[i, 5] = val_fit[i,1]
#     val_results[i, 6] = val_sig_X_COM_mean[i]
#     val_results[i, 7] = val_sig_Y_COM_mean[i]
#     val_results[i, 8] = val_X_COM_mean[i]
#     val_results[i, 9] = val_Y_COM_mean[i]
#     val_results[i, 10] = np.sqrt(((val_params[i,1] - val_fit[i,0])**2) + ((val_params[i,0] - val_fit[i,1])**2))
#     val_results[i, 11] = np.sqrt(((val_params[i,1] - val_X_COM_mean[i])**2) + ((val_params[i,0] - val_Y_COM_mean[i])**2))
#     val_results[i, 12] = np.sqrt((sig_val_fit[i,0] - val_fit[i,0])**2) + ((sig_val_fit[i,1] - val_fit[i,1])**2)
#     val_results[i, 13] = np.sqrt((val_sig_X_COM_mean[i] - val_X_COM_mean[i])**2) + ((val_sig_Y_COM_mean[i] - val_Y_COM_mean[i])**2)


# df = pd.DataFrame(results, columns = results_titles)
# df.to_csv(str(dir) +  "/results.csv")
# df = pd.DataFrame(val_results, columns = results_titles)
# df.to_csv(str(dir) +  "/val_results.csv")

# print("Testing:")
# print("fit: (", fit[0,0], fit[0,1], ") COM: (", X_COM, Y_COM, ")")
# print("true: (", params[0,1], params[0,0], "), Error:", np.sqrt(((params[0,1] - fit[0,0])**2) + ((params[0,0] - fit[0,1])**2)), "COM Error:", np.sqrt(((params[0,1] - X_COM)**2) + ((params[0,0] - Y_COM)**2)))
# print("true fit: (", sig_fit[0,0], sig_fit[0,1], "), Error:", np.sqrt((sig_fit[0,0] - fit[0,0])**2) + ((sig_fit[0,1] - fit[0,1])**2), "COM Error:", np.sqrt(((sig_X_COM - X_COM)**2) + ((sig_Y_COM - Y_COM)**2)))
# print()

# print("Validation:")
# print("fit: (", val_fit[0,0], val_fit[0,1], ") COM: (", val_X_COM, val_Y_COM, ")")
# print("true: (", val_params[0,1], val_params[0,0], "), Error:", np.sqrt(((val_params[0,1] - val_fit[0,0])**2) + ((val_params[0,0] - val_fit[0,1])**2)), "COM Error:", np.sqrt(((val_params[0,1] - val_X_COM)**2) + ((val_params[0,0] - val_Y_COM)**2)))
# print("true fit: (", sig_val_fit[0,0], sig_val_fit[0,1], "), Error:", np.sqrt(((sig_val_fit[0,0] - val_fit[0,0])**2) + ((sig_val_fit[1] - val_fit[0,1])**2)), "COM Error:", np.sqrt(((val_sig_X_COM - val_X_COM)**2) + ((val_sig_Y_COM - val_Y_COM)**2)))

# print(results_titles[:-4])
# print(results.mean(axis=0)[:-4])
# print(val_results.mean(axis=0)[:-4])

# with open(str(dir) + "/output.txt", "w+") as f:
#     print("Testing:", file=f)
#     print("fit: (", fit[0,0], fit[0,1], ") COM: (", X_COM, Y_COM, ")", file=f)
#     print("true: (", params[0,1], params[0,0], "), Error:", np.sqrt(((params[0,1] - fit[0,0])**2) + ((params[0,0] - fit[0,1])**2)), "COM Error:", np.sqrt(((params[0,1] - X_COM)**2) + ((params[0,0] - Y_COM)**2)), file=f)
#     print("true fit: (", sig_fit[0,0], sig_fit[0,1], "), Error:", np.sqrt(((sig_fit[0,0] - fit[0,0])**2) + ((sig_fit[1] - fit[0,1])**2)), "COM Error:", np.sqrt(((sig_X_COM - X_COM)**2) + ((sig_Y_COM - Y_COM)**2)), file=f)
#     print("", file=f)

#     print("Validation:", file=f)
#     print("fit: (", val_fit[0,0], val_fit[0,1], ") COM: (", val_X_COM, val_Y_COM, ")", file=f)
#     print("true: (", val_params[0,1], val_params[0,0], "), Error:", np.sqrt(((val_params[0,1] - val_fit[0,0])**2) + ((val_params[0,0] - val_fit[0,1])**2)), "COM Error:", np.sqrt(((val_params[0,1] - val_X_COM)**2) + ((val_params[0,0] - val_Y_COM)**2)), file=f)
#     print("true fit: (", sig_val_fit[0,0], sig_val_fit[1], "), Error:", np.sqrt(((sig_val_fit[0,0] - val_fit[0,0])**2) + ((sig_val_fit[0,1] - val_fit[0,1])**2)), "COM Error:", np.sqrt(((val_sig_X_COM - val_X_COM)**2) + ((val_sig_Y_COM - val_Y_COM)**2)), file=f)

#     print(results_titles[:-4], file=f)
#     print(results.mean(axis=0)[:-4], file=f)
#     print(val_results.mean(axis=0)[:-4], file=f)

