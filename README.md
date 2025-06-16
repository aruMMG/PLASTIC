# Deep Learning-Based Plastic Classification Using Spectroscopic Data

A library of deep learning models for plastic type classification using Infrared (IR) spectroscopic data. It includes models from literature (ANN, CNN) as well as our proposed ANN, CNN, and Transformer-based models designed specifically for this task. Models can be trained and tested easily by modifying YAML configuration files.

![Overview](https://github.com/aruMMG/PLASTIC/blob/main/asset/overview.jpg?raw=true)

## Model Pipeline

![Pipeline](https://github.com/aruMMG/PLASTIC/blob/main/asset/pipeline.jpg?raw=true)

The pipeline consists of two key components:
1. **Preprocessing module** (optional): Converts raw spectral input into a model-ready format.
2. **Classification model**: Predicts polymer types based on processed spectra.

## Results

| Model                | Precision | Recall | F1-Score | Time (ms) |
|---------------------|-----------|--------|----------|-----------|
| PLS_DA              | 0.390     | 0.577  | 0.455    | 0.5*      |
| LDA                 | 0.522     | 0.546  | 0.478    | 0.5*      |
| ANN1                | 0.887     | 0.832  | 0.887    | 0.1       |
| ANN2                | 0.772     | 0.902  | 0.794    | 0.1       |
| CNN                 | 0.676     | 0.756  | 0.691    | 0.5       |
| PSDN (Inception)    | 0.934     | 0.927  | 0.927    | 0.7       |
| ANN_PM              | 0.946     | 0.934  | 0.933    | 0.2       |
| **Improved_CNN_PM** | **0.984** | **0.981** | **0.981** | 1.6    |
| Transformer_PM      | 0.932     | 0.921  | 0.921    | 0.8       |

## Getting Started

```bash
git clone https://github.com/aruMMG/PLASTIC.git
cd PLASTIC
pip install -r requirements.txt
```

### Training

Prepare two YAML files (samples in `yaml/` directory):

1. **Main Config YAML**:
   - `dataset`: paths to train, test, and val directories  
   - `model`: model name (see `models/` directory)  
   - `loss`: loss function  
   - `optimizer`: optimizer settings

2. **Hyp Config YAML**:
   - Training hyperparameters

Run training:

```bash
python train.py --yaml yaml/Leone/Leone_5class_ann_pre_yaml.yaml --hyp yaml/Leone/Leone_ann_hyp.yaml
```

### Testing

Evaluate a model:

```bash
python test.py --checkpoint path/to/checkpoint.pth --yaml yaml/Leone/Leone_5class_ann_pre_yaml.yaml --hyp yaml/Leone/Leone_ann_hyp.yaml
```

## Citation

If you use this library, please cite:

```bibtex
@article{singh5191460deep,
  title={Deep Learning-Based Plastic Classification Using Spectroscopic Data},
  author={Singh, Aru Ranjan and Neo, Edward Ren Kai and Man Lai, Car and Rahman, Hasina and Hazra, Sumit and Coles, Stuart R and Peijs, Ton and Debattista, Kurt},
  journal={Available at SSRN 5191460}
}
```
