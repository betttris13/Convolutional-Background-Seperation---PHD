# Functionality to generate individual events, background and composite events then compile into a dataset.

from matplotlib import pyplot as plt
import numpy as np

# def gen_image(shape):
#     E, X, Y = shape
#     x = np.random.uniform(10, X-10)
#     y = np.random.uniform(10, Y-10)
#     r = np.random.uniform(2,4)
#     E_0 = np.random.uniform(25,50)
#     s = 1#0.01
#     alpha = np.random.uniform(0.5, 3.5)

#     # print(x, y, r, E_0, alpha)

#     x_grid, y_grid = np.meshgrid(np.linspace(0, X, Y), np.linspace(0, X, Y))

#     scale = []
#     for i in range(E):
#         A = (((i)/10+1) / 1) ** (-alpha)
#         scale.append([A])

#     scale = np.array(scale)
#     # print(E_0*scale/np.sum(scale))
#     scale = E_0*scale/scale#*scale/np.sum(scale)

#     sig = []
#     sig_clean = []
#     for i in range(20):
#         A = scale[i]
#         gaussian = A*np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * r**2)) 
#         gaussian = gaussian * np.random.poisson(2, (X,Y))
#         sig_clean.append(np.round(gaussian))
#         gaussian = np.round(gaussian)
#         sig.append(gaussian)

#     sig = np.array(sig)
#     sig_clean = np.array(sig_clean)

#     sig = sig/np.sum(sig)

#     mean = np.mean(scale)
#     # print(mean)
#     # noise = np.random.poisson(0.5, (20,50,50))
#     noise = np.random.poisson(2, (E,X,Y))
#     noise = noise/np.sum(noise)
#     obs = s*sig + noise
#     # obs = sig_clean + noise

#     sig = s*sig / obs.max(axis=(1, 2), keepdims=True)
#     sig_clean = s*sig_clean / obs.max(axis=(1, 2), keepdims=True)
#     noise = noise / obs.max(axis=(1, 2), keepdims=True)
#     obs = obs / obs.max(axis=(1, 2), keepdims=True)

#     # print(obs.max(axis=(1, 2), keepdims=True))

#     # sig = sig/np.sum(obs)
#     # noise = noise/np.sum(obs)
#     # obs = obs/np.sum(obs)

#     return obs, sig / sig.max(axis=(1, 2), keepdims=True), noise, [x, y, r]


# def make_single_set(batch, nbatch, shape):
#     obs = []
#     sig = []
#     back = []
#     params = []

#     for i in range(batch*nbatch): #9
#         if i % batch == 0:
#             print(i/batch)
#         obs_t, sig_t, back_t, params_t = gen_image(shape)
#         # print(np.sum(sig_t)/np.sum(back_t))
#         obs.append(obs_t)
#         sig.append(sig_t)
#         back.append(back_t)
#         params.append(params_t)

#     obs = np.stack([obs[i] for i in range(batch*nbatch)])
#     sig = np.stack([sig[i] for i in range(batch*nbatch)])
#     back = np.stack([back[i] for i in range(batch*nbatch)])
#     params = np.stack([params[i] for i in range(batch*nbatch)])

#     dataset = {
#         "obs": obs, 
#         "sig": sig, 
#         "back": back,
#         "params": params,
#         "events": batch*nbatch,
#         "type": "Single point-like",
#         "shape": shape
#     }

#     return dataset

def make_background_realistic(shape):
    E, X, Y = shape
    back = np.ones(shape)

    E_b = np.ones(E)
    for i in range(E):
        E_b[i] = 1.0 - 0.5*((i+1)/E) #E_b(e) = α_b(1-e)

    E_b = E_b/E_b.sum()
    back = E_b[:, np.newaxis, np.newaxis]*back

    return back

def make_point_like_realistic(shape):
    E, X, Y = shape
    x = np.random.uniform(10, X-10)
    y = np.random.uniform(10, Y-10)
    r = np.random.uniform(2, 4)
    A = np.random.uniform(0.5, 1)
    B = np.random.uniform(0.25, 0.75)

    x_grid, y_grid = np.meshgrid(np.linspace(0, X-1, X), np.linspace(0, Y-1, Y))

    sig = []
    E_s = np.ones(E)
    for i in range(E):
        E_s[i] = A*(((i)/E)**2) + B #0.75*(((i)/E)**2) + 0.25 #((0.25*i+16)/E)**2 #E_s(e) = α_se^2
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * r**2)) 
        sig.append(gaussian)

    E_s = E_s/E_s.sum()
    sig = E_s[:, np.newaxis, np.newaxis]*np.array(sig)

    return sig, [x, y, r, A, B]

def make_single_img_realistic(shape, SB):
    sig, params = make_point_like_realistic(shape)
    back = make_background_realistic(shape)

    C_S = SB*back.max()/sig.max()

    C = 10/np.mean(C_S*sig+back)
    sig = C*C_S*sig
    sig = np.random.poisson(lam=sig)
    back = C*back
    back = np.random.poisson(lam=back)
    obs = sig + back

    # Standardize -> [0,1]
    sig = sig / obs.max()
    back = back / obs.max()
    obs = obs / obs.max()

    SB_mu = sig.sum()/back.sum()
    SN_mu = sig.sum()/np.sqrt(back.sum())
    SB = sig.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)]/back.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)]
    SN = sig.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)]/np.sqrt(back.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)])
    P = sig.sum()/obs.sum()

    return obs, sig, back, params, [SB_mu, SN_mu, SB, SN, P]


def make_single_set_realistic(batch, nbatch, shape, SB):
    obs = []
    sig = []
    back = []
    params = []
    stats = []

    for i in range(batch*nbatch): #9
        if i % batch == 0:
            print(i/batch)
        obs_t, sig_t, back_t, params_t, stats_t = make_single_img_realistic(shape, SB)
        obs.append(obs_t)
        sig.append(sig_t)
        back.append(back_t)
        params.append(params_t)
        stats.append(stats_t)

    obs = np.stack([obs[i] for i in range(batch*nbatch)])
    sig = np.stack([sig[i] for i in range(batch*nbatch)])
    back = np.stack([back[i] for i in range(batch*nbatch)])
    params = np.stack([params[i] for i in range(batch*nbatch)])
    stats = np.stack([stats[i] for i in range(batch*nbatch)])

    dataset = {
        "obs": obs, 
        "sig": sig, 
        "back": back,
        "params": params,
        "events": batch*nbatch,
        "type": "Single point-like",
        "shape": shape,
        "stats": stats
    }

    return dataset

def make_multiple_img_realistic(shape, SB):
    N = np.random.uniform(1, 4)
    
    sig = np.zeros(shape)
    params = []
    for n in N:
        sig_t, params_t = make_point_like_realistic(shape)
        sig += sig_t
        params.append(params_t)
    
    back = make_background_realistic(shape)

    C_S = SB*back.max()/sig.max()

    C = 10/np.mean(C_S*sig+back)
    sig = C*C_S*sig
    sig = np.random.poisson(lam=sig)
    back = C*back
    back = np.random.poisson(lam=back)
    obs = sig + back

    # Standardize -> [0,1]
    sig = sig / obs.max()
    back = back / obs.max()
    obs = obs / obs.max()

    SB_mu = sig.sum()/back.sum()
    SN_mu = sig.sum()/np.sqrt(back.sum())
    SB = sig.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)]/back.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)]
    SN = sig.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)]/np.sqrt(back.sum(axis=0)[np.unravel_index(obs.sum(axis=0).argmax(), obs.sum(axis=0).shape)])
    P = sig.sum()/obs.sum()

    return obs, sig, back, params, [SB_mu, SN_mu, SB, SN, P]

def make_multiple_set_realistic(batch, nbatch, shape, SB):
    obs = []
    sig = []
    back = []
    params = []
    stats = []

    for i in range(batch*nbatch): #9
        if i % batch == 0:
            print(i/batch)
        obs_t, sig_t, back_t, params_t, stats_t = make_multiple_img_realistic(shape, SB)
        obs.append(obs_t)
        sig.append(sig_t)
        back.append(back_t)
        params.append(params_t)
        stats.append(stats_t)

    obs = np.stack([obs[i] for i in range(batch*nbatch)])
    sig = np.stack([sig[i] for i in range(batch*nbatch)])
    back = np.stack([back[i] for i in range(batch*nbatch)])
    params = np.stack([params[i] for i in range(batch*nbatch)])
    stats = np.stack([stats[i] for i in range(batch*nbatch)])

    dataset = {
        "obs": obs, 
        "sig": sig, 
        "back": back,
        "params": params,
        "events": batch*nbatch,
        "type": "Single point-like",
        "shape": shape,
        "stats": stats
    }

    return dataset


# def broken_power_law():