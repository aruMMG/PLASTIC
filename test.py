import os
import torch
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from Datasets.dataset import create_dataloader
from models.models import ModelFactory
from utils.utils import save_metrics, plot_confusion_matrix

def evaluate_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model configuration
    with open(args.yaml, 'r') as file:
        config = yaml.safe_load(file)

    # Dataloader
    test_path = config['test']
    class_names = config["class_names"]
    print(class_names)
    test_loader, config = create_dataloader(config=config, dir=test_path, class_names=class_names, bs=args.test_batch_size, FC=args.FC, shuffle=False)

    # Load model
    model_args = config.get('model_arguments', {})
    factory = ModelFactory()
    model = factory.load_model_from_yaml(config)

    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded model checkpoint from {args.checkpoint}")
    

    # Initialize metrics
    all_preds = []
    all_targets = []

    # Evaluate
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val = x_val.to(device)
            y_val = y_val.type(torch.LongTensor).to(device)

            outputs = model(x_val)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_val.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=3)
    print("\nClassification Report:\n", report)
    if not os.path.exists(os.path.join("logFile",args.exp_name, "test")):
        os.mkdir(os.path.join("logFile",args.exp_name, "test"))
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    np.set_printoptions(precision=3, suppress=True)  # Ensures all numpy values display 3 decimals
    print("\nConfusion Matrix:\n", cm)
    cm_figure = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
    cm_figure.figure_.savefig(os.path.join("logFile", args.exp_name, "test/confusion_matrix.png"))

    # Save metrics to file

    save_metrics(args, accuracy, report, cm, os.path.join("logFile", args.exp_name, "test/evaluation_results.txt"))

    # if args.plot_confusion_matrix:
    #     plot_confusion_matrix(cm, class_names, os.path.join("logFile", args.exp_name, "test/confusion_matrix.png"))

def update_args_with_hyp(args):
    """
    Update args with hyperparameters from a YAML file.
    Adds new arguments if they are in the YAML file but not in the argparse object.
    """
    if args.hyp:
        with open(args.hyp, 'r') as file:
            hyp_params = yaml.safe_load(file)
            for key, value in hyp_params.items():
                setattr(args, key, value)  # Add or override arguments dynamically
    return args


if __name__ == "__main__":
    """
    python evaluate_model.py \
        --exp_name my_experiment \
        --checkpoint logFile/my_experiment/weights/best_checkpoint.pth \
        --data yaml/data.yaml \
        --yaml yaml/transformer.yaml \
        --class_names PET HDPE PVC LDPE PP PS OTHER \
        --plot_confusion_matrix
    """

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained model")

    # TO RUN IN DEBUG MODE USE BELOW =====================================
    # parser.add_argument('--exp_name', type=str, default='Jeon_G1_Incep_noScheduler', help='Experiment name for logging results')
    # parser.add_argument('--checkpoint', type=str, default="logFile/Jeon_G1_Incep_noScheduler/weights/checkpoint_25.pth", help='Path to the model checkpoint')
    # parser.add_argument('--yaml', type=str, default="yaml/Jeon_PSDN_Incep.yaml", help='Path to the model configuration YAML file')
    # parser.add_argument('--hyp', type=str, default="yaml/Jeon_PSDN_Incep_hyp.yaml", help='input data directoy containing .npy files')
    # # parser.add_argument('--class_names', nargs='+', required=True, help='List of class names for the dataset')
    # =====================================================================

    # TO RUN IN DEBUG MODE USE BELOW =====================================
    parser.add_argument('--exp_name', type=str, default='inception', help='Experiment name for logging results')
    parser.add_argument('--checkpoint', type=str, default="logFile/inseption/weights/best_checkpoint.pth", help='Path to the model checkpoint')
    parser.add_argument('--yaml', type=str, default="yaml/base_yaml.yaml", help='Path to the model configuration YAML file')
    parser.add_argument('--hyp', type=str, default="yaml/base_hyp.yaml", help='input data directoy containing .npy files')
    # parser.add_argument('--class_names', nargs='+', required=True, help='List of class names for the dataset')
    # =====================================================================


    # parser.add_argument('--exp_name', type=str, default='Evaluation', help='Experiment name for logging results')
    # parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    # parser.add_argument('--yaml', type=str, required=True, help='Path to the model configuration YAML file')
    # parser.add_argument('--hyp', type=str, default='yaml/hyp.yaml', help='input data directoy containing .npy files')
    # # parser.add_argument('--class_names', nargs='+', required=True, help='List of class names for the dataset')

    args = parser.parse_args()
    args = update_args_with_hyp(args)
    evaluate_model(args)


