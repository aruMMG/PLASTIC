import os
import torch
import matplotlib.pyplot as plt

def createFolders(args):
    """Creates required forlders for GAN training"""
    exp_name = args.exp_name
    if args.resume:
        pass
    else:
        if os.path.exists(os.path.join("logFile",args.exp_name)):
            exp_num = 1
            
            while exp_num<1000:
                exp_name = args.exp_name + f"_{exp_num}"
                if os.path.exists(os.path.join("logFile",exp_name)):
                    exp_num+=1
                    continue
                else:
                    os.mkdir(os.path.join("logFile",exp_name))
                    break
        else:
            os.mkdir(os.path.join("logFile",exp_name))

        os.mkdir(os.path.join("logFile",exp_name, "weights"))
        os.mkdir(os.path.join("logFile",exp_name, "plots"))
    return exp_name

def load_checkpoint(args, device, model, optimizer, scheduler=None):
    if os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler and checkpoint['scheduler_state']:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        assert args.epochs > start_epoch, f'The number of epoch {args.epochs} required to be higher than the start epochs {start_epoch}'
        return start_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics(args, accuracy, report, cm, results_path):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"Metrics saved to {results_path}")

def save_results(predictions, output_file, class_names):
    """
    Saves the predictions to a file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for i, pred in enumerate(predictions):
            class_name = class_names[pred]
            f.write(f"Sample {i+1},{class_name}\n")
    print(f"Results saved to {output_file}")


def find_target_dim(data_len, config):
    """
    Determines the target dimension for the data based on the model type.
    :param data_len: Original data length
    :param model: Model name (e.g., "FC", "Transformer", etc.)
    :return: Target dimension length
    """
    model_name = config['model_name']
    model_args = config.get('model_arguments', {})

    if model_name == "ANN" or model_name=="ANN1" or model_name=="ANN2":
        # Fully connected models use the original dimension
        target_len = data_len
        model_args["input_size"] = target_len
    elif model_name == "Transformer":
        # Ensure data length is a multiple of 20 (patch size)
        patch_size = model_args.get('patch_size', 20)
        target_len = (data_len // patch_size) * patch_size if data_len % patch_size == 0 else ((data_len // patch_size) + 1) * patch_size
        model_args["input_size"] = target_len
    elif model_name == "PSDN Residual":
        # ResidualNet: Adjust data length for pooling layers
        alignment_factor = 20
        data_len = ((data_len + alignment_factor - 1) // alignment_factor) * alignment_factor
        target_len = max(data_len, 200)  # Minimum length for reshaping logic
    elif model_name == "CNN Inception":
         # InceptionNetwork: Adjust for pooling layers
        for pool_factor in [2, 2]:
            data_len = (data_len // pool_factor) * pool_factor
        target_len = max(data_len, 64)  # Ensures compatibility with inception logic
        model_args["input_size"] = target_len
    elif model_name == "Improved Inception":
         # InceptionNetwork: Adjust for pooling layers
        for pool_factor in [2, 2]:
            data_len = (data_len // pool_factor) * pool_factor
        target_len = max(data_len, 64)  # Ensures compatibility with inception logic
        model_args["input_size"] = target_len
    elif  model_name=="CNN":
        target_len = max(data_len, 512)
        model_args["input_size"] = target_len


    elif model_name == "PSDN Inception":
        # InceptionNetwork: Adjust for pooling layers
        for pool_factor in [2, 2]:
            data_len = (data_len // pool_factor) * pool_factor
        target_len = max(data_len, 64)  # Ensures compatibility with inception logic
        model_args["input_size"] = target_len
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    config['model_arguments'] = model_args
    return target_len, config
