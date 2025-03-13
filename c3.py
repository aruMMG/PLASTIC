
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import numpy as np
import random
# Load the data
# heatmap = np.load("logFile/Edi_MIR/baseline/Improved_Incep_pre/heatmap0.npy")  # Shape: (n, 4000)
# input_tensor = np.load("logFile/Edi_MIR/baseline/Improved_Incep_pre/input_tensors0.npy")  # Shape: (n, 4000)

def heatmap_plot(heatmap, input_tensor, output_dir, selected_indices=None):
    # Check if dimensions match
    assert heatmap.shape == input_tensor.shape, "Shapes of heatmap and input_tensor do not match"

    num_samples, num_points = input_tensor.shape
    # Select 10 random samples if there are more than 10 samples
    # Select 10 samples based on provided indices or randomly
    if selected_indices is not None:
        assert len(selected_indices) == 10, "selected_indices must have exactly 10 elements"
        assert max(selected_indices) < num_samples, "Indices must be within range of available samples"
    else:
        if num_samples > 10:
            selected_indices = random.sample(range(num_samples), 10)
            print(selected_indices)
        else:
            selected_indices = list(range(num_samples))  # Use all if ≤ 10 samples

    # Filter data for selected indices
    heatmap = heatmap[selected_indices]
    input_tensor = input_tensor[selected_indices]
    num_samples, num_points = input_tensor.shape

    # Create a directory to save plots
    os.makedirs(output_dir, exist_ok=True)

    # Define a colormap from light blue to orange
    cmap = plt.cm.cool

    # Plot each sample
    for i in range(num_samples):
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Normalize the heatmap values for the current sample
        norm = mcolors.Normalize(vmin=np.min(heatmap[i]), vmax=np.max(heatmap[i]))
        colors = cmap(norm(heatmap[i]))
        
        # Line plot using input_tensor values, colored by heatmap values
        for j in range(num_points - 1):
            ax.plot([j, j+1], [input_tensor[i, j], input_tensor[i, j+1]], color=colors[j])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Heatmap Intensity')
        
        ax.set_xlabel("Index")
        ax.set_ylabel("Input Tensor Value")
        ax.set_title(f"Sample {i+1}")
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
        plt.close()

    # Compute the mean heatmap and mean input tensor
    mean_heatmap = np.mean(heatmap, axis=0)
    mean_input_tensor = np.mean(input_tensor, axis=0)

    # Plot mean heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    # Normalize the mean heatmap values
    norm = mcolors.Normalize(vmin=np.min(mean_heatmap), vmax=np.max(mean_heatmap))
    colors = cmap(norm(mean_heatmap))

    # Line plot for the mean input tensor
    for j in range(num_points - 1):
        ax.plot([j, j+1], [mean_input_tensor[j], mean_input_tensor[j+1]], color=colors[j])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Mean Heatmap Intensity')

    ax.set_xlabel("Index")
    ax.set_ylabel("Mean Input Tensor Value")
    ax.set_title("Mean Heatmap Plot")
    
    num_ticks = 10  # You can change this to control the number of ticks
    tick_positions = np.linspace(0, num_points - 1, num_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Customize major and minor grid lines
    ax.minorticks_on()  # Enable minor ticks
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)  # Dotted minor grid

    # Save the mean plot
    plt.savefig(os.path.join(output_dir, "mean_plot.png"))
    plt.close()

    print(f"Plots saved in '{output_dir}' directory.")
    return selected_indices
# Example usage
# heatmap_plot(heatmap, input_tensor, "output_plots")
