# Script to train model

from scipy.optimize import curve_fit
import pandas as pd 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

import data_gen
from model import ConvVAE

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)  
print(device)

# ---------- Loss Function ----------

def vae_loss(recon_x, x, mu, logvar, alpha = 1.0, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    # penalty_loss = F.l1_loss(recon_x, x, reduction='sum')
    penalty_loss = torch.norm(recon_x, p=1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div + alpha * penalty_loss

def train(model, model_params, training_data):
    # ---------- Optimizer ----------
    optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])

    # ---------- Dataloader ----------
    obs = training_data["obs"]
    sig = training_data["sig"]
    params = training_data["params"]

    obs = torch.from_numpy(obs).type(torch.float)
    sig = torch.from_numpy(sig).type(torch.float)
    # back = torch.from_numpy(back).type(torch.float)
    params = torch.from_numpy(params).type(torch.float)

    data_set = TensorDataset(obs, sig, params)  # Midify to alsop have background
    train_loader = DataLoader(data_set, batch_size=model_params["batch_size"], shuffle=True)

    # ---------- Training Loop ----------
    losses = []
    epochs = model_params["epochs"]
    alpha = model_params["alpha"]
    beta = model_params["beta"]
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for input_img, target_img, target_params in progress_bar:
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            optimizer.zero_grad()

            recon_x, mu, logvar = model(input_img)
            loss = vae_loss(recon_x, target_img, mu, logvar, alpha, beta)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item() / input_img.size(0))

        avg_loss = train_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.2f}")

    return model, losses