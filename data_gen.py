# Functionality to generate individual events, background and composite events then compile into a dataset.

from matplotlib import pyplot as plt
import numpy as np

def gen_image(shape):
    E, X, Y = shape
    x = np.random.uniform(10, X-10)
    y = np.random.uniform(10, Y-10)
    r = np.random.uniform(2,4)
    E_0 = np.random.uniform(25,50)
    s = 0.01
    alpha = np.random.uniform(0.5, 3.5)

    # print(x, y, r, E_0, alpha)

    x_grid, y_grid = np.meshgrid(np.linspace(0, X, Y), np.linspace(0, X, Y))

    scale = []
    for i in range(E):
        A = (((i)/10+1) / 1) ** (-alpha)
        scale.append([A])

    scale = np.array(scale)
    # print(E_0*scale/np.sum(scale))
    scale = E_0*scale/scale#*scale/np.sum(scale)

    sig = []
    sig_clean = []
    for i in range(20):
        A = scale[i]
        gaussian = A*np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * r**2)) 
        gaussian = gaussian * np.random.poisson(2, (X,Y))
        sig_clean.append(np.round(gaussian))
        gaussian = np.round(gaussian)
        sig.append(gaussian)

    sig = np.array(sig)
    sig_clean = np.array(sig_clean)

    sig = sig/np.sum(sig)

    mean = np.mean(scale)
    # print(mean)
    # noise = np.random.poisson(0.5, (20,50,50))
    noise = np.random.poisson(2, (E,X,Y))
    noise = noise/np.sum(noise)
    obs = s*sig + noise
    # obs = sig_clean + noise

    sig = s*sig / obs.max(axis=(1, 2), keepdims=True)
    sig_clean = s*sig_clean / obs.max(axis=(1, 2), keepdims=True)
    noise = noise / obs.max(axis=(1, 2), keepdims=True)
    obs = obs / obs.max(axis=(1, 2), keepdims=True)

    # print(obs.max(axis=(1, 2), keepdims=True))

    # sig = sig/np.sum(obs)
    # noise = noise/np.sum(obs)
    # obs = obs/np.sum(obs)

    return obs, sig / sig.max(axis=(1, 2), keepdims=True), noise, [x, y, r]


def make_single_set(batch, nbatch, shape):
    obs = []
    sig = []
    back = []
    params = []

    for i in range(batch*nbatch): #9
        if i % batch == 0:
            print(i/batch)
        obs_t, sig_t, back_t, params_t = gen_image(shape)
        # print(np.sum(sig_t)/np.sum(back_t))
        obs.append(obs_t)
        sig.append(sig_t)
        back.append(back_t)
        params.append(params_t)

    obs = np.stack([obs[i] for i in range(batch*nbatch)])
    sig = np.stack([sig[i] for i in range(batch*nbatch)])
    back = np.stack([back[i] for i in range(batch*nbatch)])
    params = np.stack([params[i] for i in range(batch*nbatch)])

    dataset = {
        "obs": obs, 
        "sig": sig, 
        "back": back,
        "params": params,
        "events": batch*nbatch,
        "type": "Single point-like",
        "shape": shape
    }

    return dataset
