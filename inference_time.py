import time
import torch
import numpy as np
import yaml
from models.models import ModelFactory


def load_model(args, device):
    factory = ModelFactory()
    with open(args.yaml, 'r') as file:
        config = yaml.safe_load(file)

    model = factory.load_model_from_yaml(config)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model checkpoint from {args.checkpoint}")
    return model


def benchmark_random_input(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = load_model(args, device)

    # Dummy input shape based on your model (e.g., [B, 1, 832] for FTIR)
    input_shape = (1, 4000)
    batch_size = 512

    # Generate fixed random tensors
    random_single_inputs = torch.randn((batch_size, *input_shape[:]), device=device)
    random_batch_input = torch.randn((batch_size, *input_shape[:]), device=device)

    # --- 1. Single-spectrum inference 1000x ---
    single_times = []
    with torch.no_grad():
        for i in range(batch_size):
            sample = random_single_inputs[i].unsqueeze(0)
            start = time.time()
            _ = model(sample)
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.time()
            single_times.append(end - start)

    avg_single_time = np.mean(single_times)
    print(f"\n[Single-spectrum] Average Inference Time: {avg_single_time * 1000:.3f} ms")

    # --- 2. Batched inference 1000x on batch of 1000 ---
    batch_times = []
    with torch.no_grad():
        for _ in range(1000):
            start = time.time()
            _ = model(random_batch_input)
            # torch.cuda.synchronize() if device == "cuda" else None
            end = time.time()
            batch_times.append(end - start)

    avg_batch_time = np.mean(batch_times)
    print(f"[Batch-1000] Average Inference Time: {avg_batch_time * 1000:.3f} ms")
    print(f"[Batch-1000] Per-spectrum Time: {(avg_batch_time / batch_size) * 1000:.3f} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random Input Inference Benchmark")

    parser.add_argument('--checkpoint', default="logFile/Edi_MIR/baseline/Edi_MIR_trans_pre/Edi_MIR_trans_pre_1/weights/best_checkpoint.pth", type=str)
    parser.add_argument('--yaml', default="yaml/Edi_MIR/trans_pre/baseline/trans_pre_1.yaml", type=str)

    args = parser.parse_args()
    benchmark_random_input(args)
