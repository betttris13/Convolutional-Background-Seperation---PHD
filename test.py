# Script to test model under different inputs

from scipy.optimize import curve_fit
import pandas as pd 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import data_gen
from model import ConvVAE

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)  

def fit_brightest(img):
    E, _, _ = img.shape
    fit = []
    for i in range(20):
        fit.append(np.unravel_index(np.argmax(img[i]), img[i].shape))
    fit = np.nanmean(fit, axis=(0))
    return fit

# def _calc_COM_single(img):
#     E, X, Y = img.shape
#     X_COM = np.zeros(E)
#     Y_COM = np.zeros(E)
#     M = np.zeros(E)

#     for x in range(X):
#         for y in range(Y):
#             for e in range(E):
#                 m = img[e, x, y]
#                 X_COM[e] += x * m
#                 Y_COM[e] += y * m
#                 M[e] += m

#     M[M == 0] = 1e-8
#     return np.mean(X_COM / M), np.mean(Y_COM / M)

# def calc_COM(imgs, num_workers=16, show_progress=True):
#     """
#     Calculate center of mass for images in parallel.

#     Parameters:
#     - imgs: numpy array (N, E, X, Y)
#     - num_workers: number of parallel processes
#     - show_progress: whether to display tqdm progress bar

#     Returns:
#     - X_COM_mean: np.ndarray (N,)
#     - Y_COM_mean: np.ndarray (N,)
#     """
#     N = imgs.shape[0]
#     X_COM_mean = np.zeros(N)
#     Y_COM_mean = np.zeros(N)

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(_calc_COM_single, imgs[n]) for n in range(N)]
#         for i, future in enumerate(tqdm(as_completed(futures), total=N, disable=not show_progress, desc="Computing COM")):
#             x_com, y_com = future.result()
#             X_COM_mean[i] = x_com
#             Y_COM_mean[i] = y_com

#     return X_COM_mean, Y_COM_mean

def calc_COM(imgs):
        N, E, X, Y = imgs.shape
        X_COM_mean = np.zeros(N)
        Y_COM_mean = np.zeros(N)
        
        for n in tqdm(range(N)):
            X_COM = np.zeros(E)
            Y_COM = np.zeros(E)
            M = np.zeros(E)
            for x in range(X):
                for y in range(Y):
                    for e in range(E):
                        m = imgs[n,e,x,y]
                        X_COM[e] += x * m
                        Y_COM[e] += y * m
                        M[e] += m
        
            for e in range(E):
                X_COM[e] = X_COM[e]/M[e]
                Y_COM[e] = Y_COM[e]/M[e]
            
            X_COM_mean[n] = np.mean(X_COM)
            Y_COM_mean[n] = np.mean(Y_COM)

        return X_COM_mean, Y_COM_mean

def gaussian_2d(X, Y, r, x0, y0):
    ax = np.linspace(0, X-1, X)
    ay = np.linspace(0, Y-1, Y)
    xx, yy = np.meshgrid(ax, ay)
    
    gaussian = np.exp(-((yy-x0)**2 + (xx-y0)**2) / (2 * r**2)) #note inverted xy
    gaussian /= np.max(gaussian)

    return gaussian

def calc_squares(A, B):
    mask = A != 0
    squared_diff = (A - B)**2
    # squares = np.sum(squared_diff)
    result = np.where(mask, squared_diff, 0)
    squares = np.sum(result)
    return squares

def radius(imgs, X_mean, Y_mean):
    N, E, X, Y = imgs.shape

    stds = []

    for n in tqdm(range(N)):
        img  = np.zeros((X, Y))
        for e in range(E):
            img += (imgs[n, e]/np.max(imgs[n, e]))
        
        img /= np.max(img)
        
        step = 0.01
        std = 0
        start = 1
        stop = 4
        min = np.inf
        a = []
        while abs(step) >= 0.0001:
            pointer = start
            while pointer <= stop:
                G = gaussian_2d(X, Y, pointer, X_mean[n], Y_mean[n])
                squares = calc_squares(img, G)
                a.append([pointer, squares])
                if squares < min:
                    min = squares
                    std = pointer
                pointer += step
                
            start = std - step
            stop = std + step
            step /= 10
            if start < 1:
                start = 1
            if stop > 4:
                stop = 4

        stds.append(std)

    return stds

# def radius(imgs, X_mean, Y_mean):
#     N, E, X, Y = imgs.shape

#     stds = []

#     for n in tqdm(range(N)):
#         stds_X = []
#         stds_Y = []
#         for e in range(E):
#             for y in range(Y):
#                 temp = 0
#                 for x in range(X):
#                     temp += ((x - X_mean[n])**2) * imgs[n,e,x,y]
                
#                 if temp > 0:
#                     stds_X.append(np.sqrt(temp))
#             for x in range(X):
#                 temp = 0
#                 for y in range(Y):
#                     temp += ((y - Y_mean[n])**2) * imgs[n,e,x,y]
                
#                 if temp > 0:
#                     stds_Y.append(np.sqrt(temp))


#         std_X = np.mean(stds_X)      
#         std_Y = np.mean(stds_Y)

#         stds.append((std_X+std_Y)/2)

#     return stds


def test_single(model, model_params, training_data, testing_data):

    obs = training_data["obs"]
    sig = training_data["sig"]
    back = training_data["back"]
    params = training_data["params"]

    obs = torch.from_numpy(obs).type(torch.float)
    sig = torch.from_numpy(sig).type(torch.float)
    back = torch.from_numpy(back).type(torch.float)
    params = torch.from_numpy(params).type(torch.float)

    val_obs = testing_data["obs"]
    val_sig = testing_data["sig"]
    val_back = testing_data["back"]
    val_params = testing_data["params"]

    val_obs = torch.from_numpy(val_obs).type(torch.float)
    # val_back = torch.from_numpy(val_back).type(torch.float)

    data_set = TensorDataset(obs, sig, params)  # Midify to alsop have background
    train_loader = DataLoader(data_set, batch_size=model_params["batch_size"], shuffle=False)
    
    progress_bar = tqdm(train_loader, desc="Testing")

    sig_out_list = []
    back_out_list = []
    for input_img, target_img, target_params in progress_bar:
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        

        # sig_out_t, _, _ = model(input_img)
        sig_out_t, back_out_t, _, _ = model(input_img)
        sig_out_list.append(sig_out_t.detach().cpu())
        back_out_list.append(back_out_t.detach().cpu())


    sig_out = torch.cat(sig_out_list, dim=0)
    back_out = torch.cat(back_out_list, dim=0)
    # val_sig_out, _, _  = model.forward(val_obs.to(device))
    val_sig_out, val_back_out, _, _  = model.forward(val_obs.to(device))

    obs = obs.numpy()
    sig = sig.numpy()
    back = back.numpy()
    params = params.numpy()
    sig_out= sig_out.cpu().detach().numpy()
    # sig_out = sig_out / sig_out.max(axis=(2, 3), keepdims=True)
    back_out = back_out.cpu().detach().numpy()

    val_obs = val_obs.numpy()
    val_sig_out= val_sig_out.cpu().detach().numpy()
    # val_sig_out = val_sig_out / val_sig_out.max(axis=(2, 3), keepdims=True)
    val_back_out = val_back_out.cpu().detach().numpy()



    rep_obs = np.repeat(obs[0][np.newaxis, ...], model_params["batch_size"], axis=0)
    rep_sig_out, rep_back_out, _, _  = model.forward(torch.from_numpy(rep_obs).to(device))
    rep_sig_out= rep_sig_out.cpu().detach().numpy()
    rep_back_out = rep_back_out.cpu().detach().numpy()


    rep_val_obs = np.repeat(val_obs[0][np.newaxis, ...], model_params["batch_size"], axis=0)
    rep_val_sig_out, rep_val_back_out, _, _  = model.forward(torch.from_numpy(rep_val_obs).to(device))
    rep_val_sig_out= rep_val_sig_out.cpu().detach().numpy()
    rep_val_back_out = rep_val_back_out.cpu().detach().numpy()

    # print(losses)
    print("\nCalculating COM...")

    print("1/4")
    X_COM_mean, Y_COM_mean = calc_COM(sig_out)
    rep_X_COM_mean, rep_Y_COM_mean = calc_COM(rep_sig_out)

    print("2/4")
    sig_X_COM_mean, sig_Y_COM_mean = calc_COM(sig)

    print("3/4")
    val_X_COM_mean, val_Y_COM_mean = calc_COM(val_sig_out)
    rep_val_X_COM_mean, rep_val_Y_COM_mean = calc_COM(rep_val_sig_out)

    print("4/4")
    val_sig_X_COM_mean, val_sig_Y_COM_mean = calc_COM(val_sig)

    print("\nCalculating brightest...")

    batch = model_params["batch_size"]
    nbatch = model_params["nbatch"]
    fit = np.zeros((batch*nbatch, 2))
    sig_fit = np.zeros((batch*nbatch, 2))
    for i in tqdm(range(batch*nbatch), desc="Training set"):
        temp = fit_brightest(sig_out[i])
        fit[i, 0] = temp[0]
        fit[i, 1] = temp[1]
        temp = fit_brightest(sig[i])
        sig_fit[i, 0] = temp[0]
        sig_fit[i, 1] = temp[1]

    val_fit = np.zeros((batch, 2))
    sig_val_fit = np.zeros((batch, 2))
    for i in tqdm(range(batch), desc="Validation set"):
        temp = fit_brightest(val_sig_out[i])
        val_fit[i, 0] = temp[0]
        val_fit[i, 1] = temp[1]
        temp = fit_brightest(val_sig[i])
        sig_val_fit[i, 0] = temp[0]
        sig_val_fit[i, 1] = temp[1]

    print("\nCalculating radius")

    stds_sig = radius(sig, sig_X_COM_mean, sig_Y_COM_mean)
    stds_out = radius(sig_out, sig_X_COM_mean, sig_Y_COM_mean)
    rep_stds_out = radius(rep_sig_out, rep_X_COM_mean, rep_X_COM_mean)
    val_stds_sig = radius(val_sig, val_sig_X_COM_mean, val_sig_Y_COM_mean)
    val_stds_out = radius(val_sig_out, val_sig_X_COM_mean, val_sig_Y_COM_mean)
    rep_val_stds_out = radius(rep_val_sig_out, rep_val_X_COM_mean, rep_val_X_COM_mean)

    print(stds_sig[0], params[0])
    print(stds_out[0], params[0])
    print(val_stds_sig[0], val_params[0])
    print(val_stds_out[0], val_params[0])

    results_titles = ["True x", "True y", "Signal brightest x", "Signal brightest y", "Reconstructed brightest x", "Reconstructed brightest y", "Signal COM x", "Signal COM y", "Reconstructed COM x", "Reconstructed COM y", "Error true brightest", "Error true COM", "Error fitted brightest", "Error fitted COM", "True radius", "True fitted radius", "Reconstructed fitted radius"]
    results = np.zeros((batch*nbatch, 17))
    val_results = np.zeros((batch, 17))

    for i in range(batch*nbatch):
        results[i, 0] = params[i,1]
        results[i, 1] = params[i,0]
        results[i, 2] = sig_fit[i,0]
        results[i, 3] = sig_fit[i,1]
        results[i, 4] = fit[i,0]
        results[i, 5] = fit[i,1]
        results[i, 6] = sig_X_COM_mean[i]
        results[i, 7] = sig_Y_COM_mean[i]
        results[i, 8] = X_COM_mean[i]
        results[i, 9] = Y_COM_mean[i]
        results[i, 10] = np.sqrt(((params[i,1] - fit[i,0])**2) + ((params[i,0] - fit[i,1])**2))
        results[i, 11] = np.sqrt(((params[i,1] - X_COM_mean[i])**2) + ((params[i,0] - Y_COM_mean[i])**2))
        results[i, 12] = np.sqrt((sig_fit[i,0] - fit[i,0])**2) + ((sig_fit[i,1] - fit[i,1])**2)
        results[i, 13] = np.sqrt((sig_X_COM_mean[i] - X_COM_mean[i])**2) + ((sig_Y_COM_mean[i] - Y_COM_mean[i])**2)
        results[i, 14] = params[i,2]
        results[i, 15] = stds_sig[i]
        results[i, 16] = stds_out[i]

    for i in range(batch):
        val_results[i, 0] = val_params[i,1]
        val_results[i, 1] = val_params[i,0]
        val_results[i, 2] = sig_val_fit[i,0]
        val_results[i, 3] = sig_val_fit[i,1]
        val_results[i, 4] = val_fit[i,0]
        val_results[i, 5] = val_fit[i,1]
        val_results[i, 6] = val_sig_X_COM_mean[i]
        val_results[i, 7] = val_sig_Y_COM_mean[i]
        val_results[i, 8] = val_X_COM_mean[i]
        val_results[i, 9] = val_Y_COM_mean[i]
        val_results[i, 10] = np.sqrt(((val_params[i,1] - val_fit[i,0])**2) + ((val_params[i,0] - val_fit[i,1])**2))
        val_results[i, 11] = np.sqrt(((val_params[i,1] - val_X_COM_mean[i])**2) + ((val_params[i,0] - val_Y_COM_mean[i])**2))
        val_results[i, 12] = np.sqrt((sig_val_fit[i,0] - val_fit[i,0])**2) + ((sig_val_fit[i,1] - val_fit[i,1])**2)
        val_results[i, 13] = np.sqrt((val_sig_X_COM_mean[i] - val_X_COM_mean[i])**2) + ((val_sig_Y_COM_mean[i] - val_Y_COM_mean[i])**2)
        val_results[i, 14] = val_params[i,2]
        val_results[i, 15] = val_stds_sig[i]
        val_results[i, 16] = val_stds_out[i]

    test_output = {
        "obs": training_data["obs"], 
        "sig": sig_out, 
        "back":  back_out,
        "params": training_data["params"],
        "events": training_data["events"],
        "type": "Results",
        "shape": training_data["shape"]
    }

    val_output = {
        "obs": testing_data["obs"], 
        "sig": val_sig_out, 
        "back": val_back_out,
        "params": testing_data["params"],
        "events": testing_data["events"],
        "type": "Results",
        "shape": testing_data["shape"]
    }

    rep_test_output = {
        "obs": _, 
        "sig": rep_sig_out, 
        "back":  rep_back_out,
        "params": _,
        "events": _,
        "type": "Rep",
        "shape": _
    }

    rep_val_output = {
        "obs": _, 
        "sig": rep_val_sig_out, 
        "back":  rep_val_back_out,
        "params": _,
        "events": _,
        "type": "Rep",
        "shape": _
    }

    return results_titles, results, val_results, test_output, val_output, rep_test_output, rep_val_output

if __name__ == "__main__":
    # G = gaussian_2d(50,50,3.5,25,25)
    # plt.imshow(G)
    # plt.show()
    print(0)