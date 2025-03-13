import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.models import ModelFactory

from Datasets.dataset import create_inference_dataloader
from utils.utils import save_results

def load_model(args, device):
    """
    Loads the trained model and its checkpoint.
    """
    # Load model configuration
    factory = ModelFactory()
    with open(args.yaml, 'r') as file:
        config = yaml.safe_load(file)

    model = factory.load_model_from_yaml(config)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model checkpoint from {args.checkpoint}")
    return model


def perform_inference(args):
    """
    Performs inference on the given dataset and outputs predictions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args, device)

    # Create inference dataloader
    dataloader = create_inference_dataloader(args.dataset, bs=args.batch_size, FC=args.FC)

    # Perform inference
    results = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())

    # Save results
    save_results(results, args.output_file, args.class_names)





if __name__ == "__main__":
    """
    python inference_script.py \
    --checkpoint logFile/my_experiment/weights/best_checkpoint.pth \
    --yaml yaml/transformer.yaml \
    --dataset path/to/inference_data \
    --output_file inference_results.csv \
    --batch_size 32 \
    --class_names PET HDPE PVC LDPE PP PS OTHER \
    --FC
    """
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Model Inference Script")

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--yaml', type=str, required=True, help='Path to the model configuration YAML file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_file', type=str, default="inference_results.csv", help='File to save predictions')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--class_names', nargs='+', required=True, help='List of class names for the dataset')
    parser.add_argument('--FC', action='store_true', help='Use fully connected format for features')

    args = parser.parse_args()

    perform_inference(args)
