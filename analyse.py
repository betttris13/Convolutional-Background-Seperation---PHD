# Script to analyse and save results.print("Generating full results...")
import numpy as np
from matplotlib import pyplot as plt
import json

def run_single(model_params, train_loss, training_data, testing_data, test_output, val_output, rep_test_output, rep_val_output, results_titles, results, val_results, dir):
    plt.plot(train_loss)
    plt.savefig(dir + "/loss.png")
    plt.close()

    # plt.plot(val_losses)
    # plt.savefig(dir + "/loss_val.png")
    # plt.close()

    obs = training_data["obs"]
    sig = training_data["sig"]
    back = training_data["back"]
    params = training_data["params"]

    sig_out = test_output["sig"]
    back_out = test_output["back"]

    val_obs = testing_data["obs"]
    val_sig = testing_data["sig"]
    val_back = testing_data["back"]
    val_params = testing_data["params"]

    val_sig_out = val_output["sig"]
    val_back_out = val_output["back"]

    rep_sig_out = rep_test_output["sig"]
    rep_back_out = rep_test_output["back"]

    rep_val_sig_out = rep_val_output["sig"]
    rep_val_back_out = rep_val_output["back"]

    print("Saving images...")

    for i in range(20):
        plt.imshow(obs[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/obs_"+ str(i) +".png")
        plt.colorbar()
        plt.close()
        plt.imshow(sig[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/sig_"+ str(i) +".png")
        plt.colorbar()
        plt.close()
        plt.imshow(back[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/back_"+ str(i) +".png")
        plt.colorbar()
        plt.close()
        plt.imshow(sig[0,i]+back[0,12], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/sig_back_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig_out[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(back_out[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/back_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig_out[0,i]+back_out[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/test/sig_out_back_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(obs[0,i]-back_out[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/test/obs_back_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(obs[0,i]-sig_out[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/test/obs_sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig[0,i]-sig_out[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/test/sig_sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow((obs[0,i]-back_out[0,i])/obs[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/test/obs_back_out_per_"+ str(i) +".png")
        plt.close()
        plt.imshow((obs[0,i]-sig_out[0,i])/obs[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/test/obs_sig_out_per_"+ str(i) +".png")
        plt.close()
        plt.imshow((sig[0,i]-sig_out[0,i])/sig[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/test/sig_sig_out_per_"+ str(i) +".png")
        plt.close()

    for i in range(20):
        plt.imshow(val_obs[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/obs_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/sig_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_back[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/back_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig[0,i]+val_back[0,12], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/sig_back_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig_out[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_back_out[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/back_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig_out[0,i]+val_back_out[0,i], vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(dir + "/val/sig_out_back_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_obs[0,i]-val_back_out[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/val/obs_back_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_obs[0,i]-val_sig_out[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/val/obs_sig_out_"+ str(i) +".png")
        plt.colorbar()
        plt.close()
        plt.imshow(val_sig[0,i]-val_sig_out[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/val/sig_sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow((val_obs[0,i]-val_back_out[0,i])/obs[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/val/obs_back_out_per_"+ str(i) +".png")
        plt.close()
        plt.imshow((val_obs[0,i]-val_sig_out[0,i])/obs[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/val/obs_sig_out_per_"+ str(i) +".png")
        plt.close()
        plt.imshow((val_sig[0,i]-val_sig_out[0,i])/sig[0,i], vmin=-1, vmax=1, cmap = "seismic")
        plt.colorbar()
        plt.savefig(dir + "/val/sig_sig_out_per_"+ str(i) +".png")
        plt.close()


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
    gaussian_1 = collapsed_1.max()*np.exp(-((x - params[0, 0])**2) / (2 * params[0, 2]**2))
    gaussian_2 = collapsed_2.max()*np.exp(-((x - params[0, 1])**2) / (2 * params[0, 2]**2))


    axs[0].plot(collapsed_1, label='True signal', linestyle='-')
    axs[0].plot(out_collapsed_1, label='Reconstructed signal', color='orange', linestyle='-')
    axs[0].plot(gaussian_1, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[0].axvline(x = params[0, 0], alpha=0.6, color='grey', linestyle='--')
    axs[0].set_title('X axis profile')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(collapsed_2, label='True signal', linestyle='-')
    axs[1].plot(out_collapsed_2, label='Reconstructed signal', color='orange', linestyle='-')
    axs[1].plot(gaussian_2, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[1].axvline(x = params[0, 1], alpha=0.6, color='grey', linestyle='--')
    axs[1].set_title('Y axis profile')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(dir + "/test/test_space.png")
    plt.close()

    e = np.linspace(0, 19, 20)
    E_sig_true = params[0, 3]*(((e)/20)**2) + params[0, 4]

    E_sig = np.zeros(20)
    E_sig_out = np.zeros(20)
    for i in range(sig[0].shape[0]):
        E_sig[i] = sig[0][i, :, :].mean()
        E_sig_out[i] += sig_out[0][i, :, :].mean()

    E_sig_true = E_sig[19]*E_sig_true/E_sig_true[19]

    plt.plot(E_sig, label='True signal')
    plt.plot(E_sig_true, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    plt.plot(E_sig_out, label='Reconstructed signal')
    plt.savefig(dir + "/test/test_energy.png")
    plt.close()

    e = np.linspace(0, 19, 20)
    E_back_true = params[0, 3]*(((e)/20)**2) + params[0, 4]

    E_back = np.zeros(20)
    E_back_out = np.zeros(20)
    for i in range(back[0].shape[0]):
        E_back[i] = back[0][i, :, :].mean()
        E_back_out[i] += back_out[0][i, :, :].mean()

    E_back_true = E_back[19]*E_back_true/E_back_true[19]

    plt.plot(E_back, label='True background')
    # plt.plot(E_back_true, label='Theoretical background', alpha=0.6, color='grey', linestyle='--')
    plt.plot(E_back_out, label='Reconstructed background')
    plt.savefig(dir + "/test/test_background_energy.png")
    plt.close()

    ######
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 subplots, stacked vertically

    collapsed_1 = np.zeros(50)
    collapsed_2 = np.zeros(50)
    out_collapsed_1 = np.zeros(50)
    out_collapsed_2 = np.zeros(50)
    for x in range(val_sig[0].shape[0]):
        collapsed_1 += val_sig[0][x, :, :].sum(axis=0)
        collapsed_2 += val_sig[0][x, :, :].sum(axis=1)
        out_collapsed_1 += val_sig_out[0][x, :, :].sum(axis=0)
        out_collapsed_2 += val_sig_out[0][x, :, :].sum(axis=1)

    x = np.linspace(0, 49, 50)
    gaussian_1 = collapsed_1.max()*np.exp(-((x - val_params[0, 0])**2) / (2 * val_params[0, 2]**2))
    gaussian_2 = collapsed_2.max()*np.exp(-((x - val_params[0, 1])**2) / (2 * val_params[0, 2]**2))


    axs[0].plot(collapsed_1, label='True signal', linestyle='-')
    axs[0].plot(out_collapsed_1, label='Reconstructed signal', color='orange', linestyle='-')
    axs[0].plot(gaussian_1, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[0].axvline(x = val_params[0, 0], alpha=0.6, color='grey', linestyle='--')
    axs[0].set_title('X axis profile')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(collapsed_2, label='True signal', linestyle='-')
    axs[1].plot(out_collapsed_2, label='Reconstructed signal', color='orange', linestyle='-')
    axs[1].plot(gaussian_2, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[1].axvline(x = val_params[0, 1], alpha=0.6, color='grey', linestyle='--')
    axs[1].set_title('Y axis profile')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(dir + "/val/test_space.png")
    plt.close()

    e = np.linspace(0, 19, 20)
    E_sig_true = val_params[0, 3]*(((e)/20)**2) + val_params[0, 4]
    
    E_sig = np.zeros(20)
    E_sig_out = np.zeros(20)
    for i in range(val_sig[0].shape[0]):
        E_sig[i] = val_sig[0][i, :, :].mean()
        E_sig_out[i] += val_sig_out[0][i, :, :].mean()

    E_sig_true = E_sig[19]*E_sig_true/E_sig_true[19]

    plt.plot(E_sig, label='True signal')
    plt.plot(E_sig_true, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    plt.plot(E_sig_out, label='Reconstructed signal')
    plt.savefig(dir + "/val/test_energy.png")
    plt.close()

    e = np.linspace(0, 19, 20)
    E_back_true = val_params[0, 3]*(((e)/20)**2) + val_params[0, 4]

    E_back = np.zeros(20)
    E_back_out = np.zeros(20)
    for i in range(val_back[0].shape[0]):
        E_back[i] = val_back[0][i, :, :].mean()
        E_back_out[i] += val_back_out[0][i, :, :].mean()

    E_back_true = E_back[19]*E_back_true/E_back_true[19]

    plt.plot(E_back, label='True background')
    # plt.plot(E_back_true, label='Theoretical background', alpha=0.6, color='grey', linestyle='--')
    plt.plot(E_back_out, label='Reconstructed background')
    plt.savefig(dir + "/test/test_background_energy.png")
    plt.close()

    plt.hist(results[:,10])
    plt.savefig(dir + "/test/fit_true_error.png")
    plt.close()
    plt.hist(results[:,11])
    plt.savefig(dir + "/test/COM_true_error.png")
    plt.close()
    plt.hist(results[:,12])
    plt.savefig(dir + "/test/fit_fit_error.png")
    plt.close()
    plt.hist(results[:,13])
    plt.savefig(dir + "/test/COM_COM_error.png")
    plt.close()


    plt.hist(val_results[:,10])
    plt.savefig(dir + "/val/fit_true_error.png")
    plt.close()
    plt.hist(val_results[:,11])
    plt.savefig(dir + "/val/COM_true_error.png")
    plt.close()
    plt.hist(val_results[:,12])
    plt.savefig(dir + "/val/fit_fit_error.png")
    plt.close()
    plt.hist(val_results[:,13])
    plt.savefig(dir + "/val/COM_COM_error.png")
    plt.close()

    print("Testing:")
    print("fit: (", results[0,4], results[0,5], ") COM: (", results[0,8], results[0,9], ")")
    print("true: (", results[0,0], results[0,1], "), Error:", results[0,10], "COM Error:", results[0,11])
    print("true fit: (", results[0,2], results[0,3], "), true COM: (", results[0,6], results[0,7], "), Error:", results[0,12], "COM Error:", results[0,13])
    print()

    print("Validation:")
    print("fit: (", val_results[0,4], val_results[0,5], ") COM: (", val_results[0,8], val_results[0,9], ")")
    print("true: (", val_results[0,0], val_results[0,1], "), Error:", val_results[0,10], "COM Error:", val_results[0,11])
    print("true fit: (", val_results[0,2], val_results[0,3], "), true COM: (", val_results[0,6], val_results[0,7], "), Error:", val_results[0,12], "COM Error:", val_results[0,13])
    print()

    print(results_titles[10:14])
    print("Testing:")
    print("Mean:", results.mean(axis=0)[10:14])
    print("Median:", np.median(results, axis=0)[10:14])
    print("68%:", np.quantile(results, 0.68, axis=0)[10:14])
    print("95%:", np.quantile(results, 0.95, axis=0)[10:14])
    print("Validation:")
    print("Mean:", val_results.mean(axis=0)[10:14])
    print("Median:", np.median(val_results, axis=0)[10:14])
    print("68%:", np.quantile(val_results, 0.68, axis=0)[10:14])
    print("95%:", np.quantile(val_results, 0.95, axis=0)[10:14])
    print()

    print("Testing Radius (True, True fit, Reco fit):", results[0,14], results[0,15], results[0,16])
    print("Validation Radius (True, True fit, Reco fit):", val_results[0,14], val_results[0,15], val_results[0,16])

    print(np.mean(training_data["stats"], axis = 0))


    with open(dir + "/output.txt", "w+") as f:
        print("Model params:", file=f)
        print(json.dumps(model_params, indent = 4), file=f)
        print("", file=f)

        print("Testing:", file=f)
        print("fit: (", results[0,4], results[0,5], ") COM: (", results[0,8], results[0,9], ")", file=f)
        print("true: (", results[0,0], results[0,1], "), Error:", results[0,10], "COM Error:", results[0,11], file=f)
        print("true fit: (", results[0,2], results[0,3], "), true COM: (", results[0,6], results[0,7], "), Error:", results[0,12], "COM Error:", results[0,13], file=f)
        print("", file=f)

        print("Validation:", file=f)
        print("fit: (", val_results[0,4], val_results[0,5], ") COM: (", val_results[0,8], val_results[0,9], ")", file=f)
        print("true: (", val_results[0,0], val_results[0,1], "), Error:", val_results[0,10], "COM Error:", val_results[0,11], file=f)
        print("true fit: (", val_results[0,2], val_results[0,3], "), true COM: (", val_results[0,6], val_results[0,7], "), Error:", val_results[0,12], "COM Error:", val_results[0,13], file=f)
        print("", file=f)

        print(results_titles[10:14], file=f)
        print("Testing:", file=f)
        print("Mean:", results.mean(axis=0)[10:14], file=f)
        print("Median:", np.median(results, axis=0)[10:14], file=f)
        print("68%:", np.quantile(results, 0.68, axis=0)[10:14], file=f)
        print("95%:", np.quantile(results, 0.95, axis=0)[10:14], file=f)
        print("Validation:", file=f)
        print("Mean:", val_results.mean(axis=0)[10:14], file=f)
        print("Median:", np.median(val_results, axis=0)[10:14], file=f)
        print("68%:", np.quantile(val_results, 0.68, axis=0)[10:14], file=f)
        print("95%:", np.quantile(val_results, 0.95, axis=0)[10:14], file=f)
        print("", file=f)

        print("Testing Radius (True, True fit, Reco fit):", results[0,14], results[0,15], results[0,16], file=f)
        print("Validation Radius (True, True fit, Reco fit):", val_results[0,14], val_results[0,15], val_results[0,16], file=f)




    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 subplots, stacked vertically

    collapsed_1 = np.zeros(50)
    collapsed_2 = np.zeros(50)
    out_collapsed_1 = np.zeros((64, 50))
    out_collapsed_2 = np.zeros((64, 50))
    for x in range(sig[0].shape[0]):
        collapsed_1 += sig[0][x, :, :].sum(axis=0)
        collapsed_2 += sig[0][x, :, :].sum(axis=1)
        for i in range(64):
            out_collapsed_1[i] += rep_sig_out[i][x, :, :].sum(axis=0)
            out_collapsed_2[i] += rep_sig_out[i][x, :, :].sum(axis=1)

    max_1 = np.zeros(50)
    min_1 = np.zeros(50)
    max_2 = np.zeros(50)
    min_2 = np.zeros(50)
    mean_1 = np.zeros(50)
    mean_2 = np.zeros(50)

    for i in range(50):
        max_1[i] = np.max(out_collapsed_1[:, i])
        max_2[i] = np.max(out_collapsed_2[:, i])
        min_1[i] = np.min(out_collapsed_1[:, i])
        min_2[i] = np.min(out_collapsed_2[:, i])
        mean_1[i] = np.mean(out_collapsed_1[:, i])
        mean_2[i] = np.mean(out_collapsed_2[:, i])

    x = np.linspace(0, 49, 50)
    gaussian_1 = collapsed_1.max()*np.exp(-((x - params[0, 0])**2) / (2 * params[0, 2]**2))
    gaussian_2 = collapsed_2.max()*np.exp(-((x - params[0, 1])**2) / (2 * params[0, 2]**2))


    axs[0].plot(collapsed_1, label='True signal', linestyle='-')
    axs[0].plot(mean_1, label='Reconstructed signal', color='orange', linestyle='-')
    axs[0].plot(gaussian_1, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[0].axvline(x = params[0, 0], alpha=0.6, color='grey', linestyle='--')
    axs[0].fill_between(x, min_1, max_1, alpha=0.5, color='orange')
    axs[0].set_title('X axis profile')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(collapsed_2, label='True signal', linestyle='-')
    axs[1].plot(mean_2, label='Reconstructed signal', color='orange', linestyle='-')
    axs[1].plot(gaussian_2, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[1].axvline(x = params[0, 1], alpha=0.6, color='grey', linestyle='--')
    axs[1].fill_between(x, min_2, max_2, alpha=0.5, color='orange')
    axs[1].set_title('Y axis profile')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(dir + "/test.png")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 subplots, stacked vertically

    collapsed_1 = np.zeros(50)
    collapsed_2 = np.zeros(50)
    out_collapsed_1 = np.zeros((64, 50))
    out_collapsed_2 = np.zeros((64, 50))
    for x in range(val_sig[0].shape[0]):
        collapsed_1 += val_sig[0][x, :, :].sum(axis=0)
        collapsed_2 += val_sig[0][x, :, :].sum(axis=1)
        for i in range(64):
            out_collapsed_1[i] += rep_val_sig_out[i][x, :, :].sum(axis=0)
            out_collapsed_2[i] += rep_val_sig_out[i][x, :, :].sum(axis=1)

    max_1 = np.zeros(50)
    min_1 = np.zeros(50)
    max_2 = np.zeros(50)
    min_2 = np.zeros(50)
    mean_1 = np.zeros(50)
    mean_2 = np.zeros(50)

    for i in range(50):
        max_1[i] = np.max(out_collapsed_1[:, i])
        max_2[i] = np.max(out_collapsed_2[:, i])
        min_1[i] = np.min(out_collapsed_1[:, i])
        min_2[i] = np.min(out_collapsed_2[:, i])
        mean_1[i] = np.mean(out_collapsed_1[:, i])
        mean_2[i] = np.mean(out_collapsed_2[:, i])

    x = np.linspace(0, 49, 50)
    gaussian_1 = collapsed_1.max()*np.exp(-((x - val_params[0, 0])**2) / (2 * val_params[0, 2]**2))
    gaussian_2 = collapsed_2.max()*np.exp(-((x - val_params[0, 1])**2) / (2 * val_params[0, 2]**2))


    axs[0].plot(collapsed_1, label='True signal', linestyle='-')
    axs[0].plot(mean_1, label='Reconstructed signal', color='orange', linestyle='-')
    axs[0].plot(gaussian_1, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[0].axvline(x = val_params[0, 0], alpha=0.6, color='grey', linestyle='--')
    axs[0].fill_between(x, min_1, max_1, alpha=0.5, color='orange')
    axs[0].set_title('X axis profile')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(collapsed_2, label='True signal', linestyle='-')
    axs[1].plot(mean_2, label='Reconstructed signal', color='orange', linestyle='-')
    axs[1].plot(gaussian_2, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    axs[1].axvline(x = val_params[0, 1], alpha=0.6, color='grey', linestyle='--')
    axs[1].fill_between(x, min_2, max_2, alpha=0.5, color='orange')
    axs[1].set_title('Y axis profile')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(dir + "/val.png")
    plt.close()


    e = np.linspace(0, 19, 20)
    E_sig_true = params[0, 3]*(((e)/20)**2) + params[0, 4]

    E_sig = np.zeros(20)
    E_sig_out = np.zeros((64,20))
    for i in range(sig[0].shape[0]):
        E_sig[i] = sig[0][i, :, :].mean()
        for j in range(64):
            E_sig_out[j, i] += rep_sig_out[j][i, :, :].mean()

    E_sig_true = E_sig[19]*E_sig_true/E_sig_true[19]

    max = np.zeros(20)
    min = np.zeros(20)
    mean = np.zeros(20)

    for i in range(20):
        max[i] = np.max(E_sig_out[:, i])
        min[i] = np.max(E_sig_out[:, i])
        mean[i] = np.min(E_sig_out[:, i])

    plt.plot(E_sig, label='True signal')
    plt.plot(E_sig_true, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    plt.plot(mean, label='Reconstructed signal')
    plt.fill_between(e, min, max, alpha=0.5, color='orange')
    plt.savefig(dir + "/test_energy.png")
    plt.close()

    e = np.linspace(0, 19, 20)
    E_val_sig_true = val_params[0, 3]*(((e)/20)**2) + val_params[0, 4]

    E_val_sig = np.zeros(20)
    E_val_sig_out = np.zeros((64,20))
    for i in range(sig[0].shape[0]):
        E_val_sig[i] = sig[0][i, :, :].mean()
        for j in range(64):
            E_val_sig_out[j, i] += rep_val_sig_out[j][i, :, :].mean()

    E_val_sig_true = E_val_sig[19]*E_val_sig_true/E_val_sig_true[19]

    max = np.zeros(20)
    min = np.zeros(20)
    mean = np.zeros(20)

    for i in range(20):
        max[i] = np.max(E_val_sig_out[:, i])
        min[i] = np.max(E_val_sig_out[:, i])
        mean[i] = np.min(E_val_sig_out[:, i])

    plt.plot(E_val_sig, label='True signal')
    plt.plot(E_val_sig_true, label='Theoretical signal', alpha=0.6, color='grey', linestyle='--')
    plt.plot(mean, label='Reconstructed signal')
    plt.fill_between(e, min, max, alpha=0.5, color='orange')
    plt.savefig(dir + "/val_energy.png")
    plt.close()