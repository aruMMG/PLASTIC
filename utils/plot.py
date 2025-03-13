
import torch
import numpy as np
from matplotlib import pyplot as plt

def plot_wrong(args,class_names, model, test_dataloader, weights):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    wrong_array = None
    wrong_name = None
    for x_test, y_test in test_dataloader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)                        
            
            if not (predicted == y_test).sum().item()==1:               
                if wrong_array is None:
                    wrong_array = x_test.cpu().detach().numpy().reshape(-1)
                    wrong_name = np.array([class_names[y_test[0]]+class_names[predicted[0]]])
                else:
                    wrong_array = np.vstack((wrong_array, x_test.cpu().detach().numpy().reshape(-1)))
                    wrong_name = np.concatenate((wrong_name, np.array([class_names[y_test[0]]+class_names[predicted[0]]])))
    if wrong_array is not None:
        print(wrong_array.shape)
        if len(wrong_array.shape)==1:
            wrong_array = wrong_array.reshape(1, -1)
        plot16Data(wrong_array, f"logFile/{args.exp_name}/plots/FP", name=wrong_name)

def plot16Data(x, savename, label=None, name=None):
    y = np.arange(4000)
    c=0
    for i in np.arange(0,len(x),16):
        r=-1
        c+=1
        fig, ax = plt.subplots(4,4,figsize=(15,15))
        for j in range(16):
            if j%4==0:
                r+=1
            if (i+j)<len(x):
                print(len(x[i+j,:]))
                ax[r,j%4].plot(np.arange(len(x[i+j,:])),x[i+j,:])
            else:
                break
            if isinstance(name, np.ndarray):
                ind = i+j
                ax[r,j%4].text(0.1,np.max(x[i+j,:]), str(name[ind]))
        fig.savefig(savename+str(c)+".jpg")

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(directory, save_path="plot.png"):
    """
    Reads .npy files from the given directory and plots them as line plots.
    
    Parameters:
        directory (str): Path to the directory containing .npy files.
        save_path (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(10, 6))  # Set figure size
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]  # Get all .npy files

    if not files:
        print("No .npy files found in the directory.")
        return
    
    for file in files:
        file_path = os.path.join(directory, file)
        data = np.load(file_path)  # Load .npy file
        mean_signal = np.mean(data, axis=0)  # Average over all samples for a representative plot
        
        label = os.path.splitext(file)[0]  # Use filename as label
        plt.plot(mean_signal, label=label)  # Plot the mean of all samples

    plt.xlabel("Time Steps")
    plt.ylabel("Signal Amplitude")
    plt.title("Plot of .npy Files")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot

def plot_npy_from_splits(file_name, root_directory, save_path="split_plot.png"):
    """
    Reads the same .npy file from train, val, and test folders inside root_directory and plots them.

    Parameters:
        file_name (str): The name of the .npy file to read (e.g., 'PP.npy').
        root_directory (str): The root directory containing 'train', 'val', and 'test' folders.
        save_path (str): Path to save the generated plot image.
    """
    splits = ["train", "val", "test"]
    plt.figure(figsize=(10, 6))

    for split in splits:
        file_path = os.path.join(root_directory, split, file_name)

        if os.path.exists(file_path):
            data = np.load(file_path)  # Load .npy file
            mean_signal = np.mean(data, axis=0)  # Average over all samples for a representative plot
            plt.plot(mean_signal, label=f"{split.capitalize()}")  # Plot the mean of all samples
        else:
            print(f"Warning: {file_path} not found!")

    plt.xlabel("Time Steps")
    plt.ylabel("Signal Amplitude")
    plt.title(f"Plot of {file_name} from Train, Val, and Test")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
def plot_npy_from_splits_stacked(file_name, root_directory, save_path="split_plot.png", offset=0.8):
    """
    Reads the same .npy file from train, val, and test folders inside root_directory and plots them 
    with separate y-axis offsets to avoid overlap.

    Parameters:
        file_name (str): The name of the .npy file to read (e.g., 'PP.npy').
        root_directory (str): The root directory containing 'train', 'val', and 'test' folders.
        save_path (str): Path to save the generated plot image.
        offset (int): Vertical spacing between plots to prevent overlap.
    """
    splits = ["train", "val", "test"]
    plt.figure(figsize=(10, 6))

    tick_positions = []  # Store shifted positions for setting y-ticks
    tick_labels = [] 
    # tick_Values = [0.0, 0.05,0.1, 0.15]
    tick_Values = [0.0, 0.2, 0.4, 0.6]
    # tick_Values = [0.0, 0.10, 0.20]
    for idx, split in enumerate(splits):
        file_path = os.path.join(root_directory, split, file_name)

        if os.path.exists(file_path):
            data = np.load(file_path)  # Load .npy file
            mean_signal = np.mean(data, axis=0)  # Average over all samples for a representative plot
            plt.plot(mean_signal + idx * offset, label=f"{split.capitalize()}")  # Apply vertical shift
            shift = idx * offset
            plt.axhline(y=shift, color="gray", linestyle="-", alpha=0.6)
            for tick in tick_Values:
                tick_positions.append(shift + tick)
                tick_labels.append(f"{tick:.2f}")  # Keep label as 0.00, 0.25, 0.50
        else:
            print(f"Warning: {file_path} not found!")

    # plt.xlabel("Time Steps")
    # plt.ylabel("Signal Amplitude (Shifted)")
    # plt.title(f"Stacked Plot of {file_name} from Train, Val, and Test")
    # plt.yticks([])
    plt.yticks(tick_positions, tick_labels)
    # plt.xticks([])
    # plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_multiple_csvs(csv_files, save_path="step_value_plot.png"):
    """
    Reads multiple CSV files, extracts 'Step' and 'Value' columns, and plots them side-by-side.
    Each plot will have independent y-axis limits.

    Parameters:
        csv_files (list): List of CSV file paths (must be exactly 3 files).
        save_path (str): Path to save the generated plot image.
    """
    if len(csv_files) != 3:
        print("Error: Please provide exactly 3 CSV files.")
        return

    # Create a figure with 3 columns in one row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Training Loss", "Val Loss", "Val Accuracy"]
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)

        # Ensure required columns exist
        if "Step" not in df.columns or "Value" not in df.columns:
            print(f"Error: {csv_file} must contain 'Step' and 'Value' columns.")
            return

        # Plot line plot (no markers)
        axes[i].plot(df["Step"], df["Value"], linestyle="-", label=f"Plot {i+1}")

        # Formatting
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("Iterations")
        # axes[i].set_title(f"Plot {i+1}")
        axes[i].grid(True)
        # axes[i].legend()

    # Set y-axis label only for the first subplot
    # axes[0].set_ylabel("Value")

    plt.tight_layout()  # Adjust layout to fit
    plt.savefig(save_path)  # Save the plot
    # plt.show()

# Example usage:
# plot_multiple_csvs(["data1.csv", "data2.csv", "data3.csv"], save_path="step_value_plot.png")



if __name__=="__main__":
    csv_files = ["/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_1_Train_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_1_Val_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_1_Val_Accuracy.csv"]
    plot_multiple_csvs(csv_files, save_path="FTIR_trans_pre_1.png")
    csv_files = ["/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_2_Train_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_2_Val_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_2_Val_Accuracy.csv"]
    plot_multiple_csvs(csv_files, save_path="FTIR_trans_pre_2.png")
    csv_files = ["/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_3_Train_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_3_Val_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_3_Val_Accuracy.csv"]
    plot_multiple_csvs(csv_files, save_path="FTIR_trans_pre_3.png")
    csv_files = ["/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_4_Train_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_4_Val_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_4_Val_Accuracy.csv"]
    plot_multiple_csvs(csv_files, save_path="FTIR_trans_pre_4.png")
    csv_files = ["/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_5_Train_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_5_Val_loss.csv",
                "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/Downloads/FTIR_trans_pre_5_Val_Accuracy.csv"]
    plot_multiple_csvs(csv_files, save_path="FTIR_trans_pre_5.png")
# Example usage:
    # plot_npy_files("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="npy_plot.png")
    # plot_npy_from_splits("PE.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="split_plot.png")
    # plot_npy_from_splits_stacked("PP.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PP.png")
    # plot_npy_from_splits_stacked("PE.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PE.png")
    # plot_npy_from_splits_stacked("PET.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PET.png")
    # plot_npy_from_splits_stacked("PVC.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PVC.png")
    # plot_npy_from_splits_stacked("PS.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PS.png")
    # plot_npy_from_splits_stacked("ABS.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="ABS.png")
    # plot_npy_from_splits_stacked("PC.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PC.png")
    # plot_npy_from_splits_stacked("PMMA.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="PMMA.png")
    # plot_npy_from_splits_stacked("Others.npy", "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed/", save_path="Others.png")