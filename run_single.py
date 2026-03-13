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
import train
import test
import analyse
import webhook
import yaml

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  

device = torch.device(dev)  
print(device)


# obs = torch.from_numpy(obs).type(torch.float)
# sig = torch.from_numpy(sig).type(torch.float)
# back = torch.from_numpy(back).type(torch.float)
# params = torch.from_numpy(params).type(torch.float)
# val_obs = torch.from_numpy(val_obs).type(torch.float)
# val_sig = torch.from_numpy(val_sig).type(torch.float)
# val_back = torch.from_numpy(val_back).type(torch.float)

# ---------- Hyperparameters ----------


# Load nested YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access the nested dictionary
model_params = config["model_params"]

# model_params = {
#     "batch_size": 64,
#     "nbatch": 400,
#     "latent_dim": 512,#512,
#     "epochs": 400,
#     "learning_rate": 1e-4,
#     "alpha": 0.05, #0.05 0.1
#     "beta": 0.0, # 0.00 0.4
#     "debug": False,
#     "SB": 0.1
# }

dir = 1
while os.path.isdir(str(dir)):
    dir += 1
dir = str(dir)

# OVERRIDE dir
# dir = -1

if os.path.isdir(dir) == False:
    os.makedirs(dir)
    os.makedirs(dir + "/test")
    os.makedirs(dir + "/val")

# ---------- Model ----------
model = ConvVAE(in_channels=20, latent_dim=model_params["latent_dim"]).to(device)

webhook.send("https://tenor.com/view/ah-shit-here-we-go-gif-15190379")
# try:
if model_params["data"] == -1:
    webhook.send("Hey Lettie, your code is generating data.")
    training_data = data_gen.make_single_set_realistic(model_params["batch_size"], model_params["nbatch"], (20,50,50), model_params["SB"])
    testing_data = data_gen.make_single_set_realistic(model_params["batch_size"], 1, (20,50,50), model_params["SB"])
    with open(f'{dir}/training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    with open(f'{dir}/val_data.pkl', 'wb') as f:
        pickle.dump(testing_data, f)
else:
    with open(f'{model_params["data"]}/training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    with open(f'{model_params["data"]}/val_data.pkl', 'rb') as f:
        testing_data = pickle.load(f)

webhook.send("And now its training.")
model, train_loss = train.train(model, model_params, training_data)
torch.save(model.state_dict(), dir + '/model_weights.pth')

# If running in debug mode load model. This allows for interrupted runs to be analysed or runs to skip training etc. Should manually override dir variable to do so.
if model_params["debug"]:
    model.load_state_dict(torch.load(dir + 'model_weights.pth'))

webhook.send("Its running tests now, surprised it got this far...")
results_titles, results, val_results, test_output, val_output, rep_test_output, rep_val_output = test.test_single(model, model_params, training_data, testing_data)

df = pd.DataFrame(results, columns = results_titles)
df.to_csv(dir +  "/results.csv")
df = pd.DataFrame(val_results, columns = results_titles)
df.to_csv(dir +  "/val_results.csv")

with open(f'{dir}/test_output.pkl', 'wb') as f:
    pickle.dump(test_output, f)

with open(f'{dir}/val_output.pkl', 'wb') as f:
    pickle.dump(val_output, f)

webhook.send("Hey it made it to analysis, maybe you didn't fuck up this time?")
analyse.run_single(model_params, train_loss, training_data, testing_data, test_output, val_output, rep_test_output, rep_val_output, results_titles, results, val_results, dir)

webhook.send("Hey look everyone, Lettie's code finished. She actually didn't fuck up for one...")

# except Exception as e: 
#     print(e)
#     webhook.send("WARNING: Code failed to finish.")