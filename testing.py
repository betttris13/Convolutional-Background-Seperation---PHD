import numpy as np
from matplotlib import pyplot as plt
from data_gen import make_single_set_realistic

data = make_single_set_realistic(2, 2, (20,50,50), 0.8)
data_out = make_single_set_realistic(2, 2, (20,50,50), 0.8)

sig = data["sig"]
sig_out = data_out["sig"]

# # Prepare the plot
# plt.figure(figsize=(10, 6))

# collapsed = np.zeros(50)
# for x in range(sig[0].shape[0]):
#     collapsed += sig[0][x, :, :].sum(axis=1)
    
# # print(collapsed.shape)

# plt.plot(collapsed, label=f'x={x}', alpha=0.3)  # alpha to reduce clutter

# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.close()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 subplots, stacked vertically

collapsed_1 = np.zeros(50)
collapsed_2 = np.zeros(50)
out_collapsed_1 = np.zeros(50)
out_collapsed_2 = np.zeros(50)
for x in range(sig[0].shape[0]):
    collapsed_1 += sig[0][x, :, :].sum(axis=0)
    collapsed_2 += sig[0][x, :, :].sum(axis=1)
    out_collapsed_1 += sig_out[0][x, :, :].sum(axis=0)
    out_collapsed_2 += sig_out[0][x, :, :].sum(axis=1)

x = np.linspace(0, 49, 50)
gaussian_1 = collapsed_1.max()*np.exp(-((x - data["params"][0, 0])**2) / (2 * data["params"][0, 2]**2))
gaussian_2 = collapsed_2.max()*np.exp(-((x - data["params"][0, 1])**2) / (2 * data["params"][0, 2]**2))

axs[0].plot(collapsed_1, label='True signal', linestyle='-')
axs[0].plot(out_collapsed_1, label='Reconstructed signal', color='orange', linestyle='-')
axs[0].plot(gaussian_1, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
axs[0].axvline(x = data["params"][0, 0], alpha=0.6, color='grey', linestyle='--')
axs[0].set_title('X axis profile')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(collapsed_2, label='True signal', linestyle='-')
axs[1].plot(out_collapsed_2, label='Reconstructed signal', color='orange', linestyle='-')
axs[1].plot(gaussian_2, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
axs[1].axvline(x = data["params"][0, 1], alpha=0.6, color='grey', linestyle='--')
axs[1].set_title('Y axis profile')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
plt.close()


e = np.linspace(0, 19, 20)
E_sig_true = data["params"][0, 3]*(((e)/20)**2) + data["params"][0, 4]

E_sig = np.zeros(20)
E_sig_out = np.zeros(20)
for i in range(sig[0].shape[0]):
    E_sig[i] = sig[0][i, :, :].mean()
    E_sig_out[i] += sig_out[0][i, :, :].mean()

E_sig_true = E_sig[0]*E_sig_true/E_sig_true[0]

plt.plot(E_sig, label='True signal')
plt.plot(E_sig_true, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
plt.plot(E_sig_out, label='Reconstructed signal')
plt.legend()
plt.show()