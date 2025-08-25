# Script to analyse and save results.print("Generating full results...")
import numpy as np
from matplotlib import pyplot as plt
import json

def run_single(model_params, train_loss, training_data, testing_data, test_output, val_output, results_titles, results, val_results, dir):
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
    val_params = training_data["params"]

    val_sig_out = val_output["sig"]
    val_back_out = val_output["back"]

    print("Saving images...")

    for i in range(20):
        plt.imshow(obs[0,i])
        plt.savefig(dir + "/test/obs_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig[0,i])
        plt.savefig(dir + "/test/sig_"+ str(i) +".png")
        plt.close()
        plt.imshow(back[0,i])
        plt.savefig(dir + "/test/back_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig[0,i]+back[0,12])
        plt.savefig(dir + "/test/sig_back_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig_out[0,i])
        plt.savefig(dir + "/test/sig_out_"+ str(i) +".png")
        plt.close()
        # plt.imshow(back_out[0,i])
        # plt.savefig(dir + "/test/back_out_"+ str(i) +".png")
        # plt.close()
        # plt.imshow(sig_out[0,i]+back_out[0,i])
        # plt.savefig(dir + "/test/sig_out_back_out_"+ str(i) +".png")
        # plt.close()
        # plt.imshow(obs[0,i]-back_out[0,i])
        # plt.savefig(dir + "/test/obs_back_out_"+ str(i) +".png")
        # plt.close()
        plt.imshow(obs[0,i]-sig_out[0,i])
        plt.savefig(dir + "/test/obs_sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(sig[0,i]-sig_out[0,i])
        plt.savefig(dir + "/test/sig_sig_out_"+ str(i) +".png")
        plt.close()

    for i in range(20):
        plt.imshow(val_obs[0,i])
        plt.savefig(dir + "/val/obs_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig[0,i])
        plt.savefig(dir + "/val/sig_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_back[0,i])
        plt.savefig(dir + "/val/back_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig[0,i]+val_back[0,12])
        plt.savefig(dir + "/val/sig_back_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig_out[0,i])
        plt.savefig(dir + "/val/sig_out_"+ str(i) +".png")
        plt.close()
        # plt.imshow(val_back_out[0,i])
        # plt.savefig(dir + "/val/back_out_"+ str(i) +".png")
        # plt.close()
        # plt.imshow(val_sig_out[0,i]+val_back_out[0,i])
        # plt.savefig(dir + "/val/sig_out_back_out_"+ str(i) +".png")
        # plt.close()
        # plt.imshow(val_obs[0,i]-val_back_out[0,i])
        # plt.savefig(dir + "/val/obs_back_out_"+ str(i) +".png")
        # plt.close()
        plt.imshow(val_obs[0,i]-val_sig_out[0,i])
        plt.savefig(dir + "/val/obs_sig_out_"+ str(i) +".png")
        plt.close()
        plt.imshow(val_sig[0,i]-val_sig_out[0,i])
        plt.savefig(dir + "/val/sig_sig_out_"+ str(i) +".png")
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

        print(results_titles[10:13], file=f)
        print("Testing:", file=f)
        print("Mean:", results.mean(axis=0)[10:13], file=f)
        print("Median:", np.median(results, axis=0)[10:13], file=f)
        print("68%:", np.quantile(results, 0.68, axis=0)[10:13], file=f)
        print("95%:", np.quantile(results, 0.95, axis=0)[10:13], file=f)
        print("Validation:", file=f)
        print("Mean:", val_results.mean(axis=0)[10:13], file=f)
        print("Median:", np.median(val_results, axis=0)[10:13], file=f)
        print("68%:", np.quantile(val_results, 0.68, axis=0)[10:13], file=f)
        print("95%:", np.quantile(val_results, 0.95, axis=0)[10:13], file=f)
        print("", file=f)

        print("Testing Radius (True, True fit, Reco fit):", results[0,14], results[0,15], results[0,16], file=f)
        print("Validation Radius (True, True fit, Reco fit):", val_results[0,14], val_results[0,15], val_results[0,16], file=f)
