"""
Loss Function Documentation
----------------------------

This module provides implementations of various loss functions, along with 
background theory, mathematical formulations, and practical usage guidance 
to help users select the appropriate loss function for their tasks.

---

Loss Functions
--------------

1. **Cross-Entropy Loss**

   Description:
   - Measures the dissimilarity between the predicted probability distribution 
     and the true label distribution. It is widely used for classification tasks.

   Mathematics:
       L = -sum(y_true * log(y_pred))

   Where:
   - `y_true`: True labels (one-hot encoded).
   - `y_pred`: Predicted probabilities from the model.

   Implementation:
   - Available as `torch.nn.CrossEntropyLoss`.

   Parameters:
   - `weight`: A manual rescaling weight for each class, used to handle class imbalance.

   Use Cases:
   - Multi-class classification tasks.

---

2. **Focal Loss**

   Description:
   - Extends Cross-Entropy Loss to focus more on hard-to-classify examples.
   - Useful for handling class imbalance in tasks like object detection.

   Mathematics:
       FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

   Where:
   - `p_t`: Predicted probability of the true class.
   - `alpha`: Scaling factor for balancing classes.
   - `gamma`: Focusing parameter to emphasize harder examples.

   Implementation:
   - Provided as a custom class `FocalLoss`.

   Parameters:
   - `gamma`: Controls the focusing effect (default: 2.0).
   - `alpha`: Balancing factor for classes (default: None).
   - `reduction`: Specifies reduction mode ('mean', 'sum', or 'none').

   Use Cases:
   - Tasks with severe class imbalance (e.g., object detection).

---

Dynamic Loss Function Selection
-------------------------------

The `get_loss` function allows dynamic selection of loss functions based on their name and arguments. 
This enables users to flexibly choose loss functions as per their task requirements.

Supported Loss Functions:
- `cross_entropy`: Standard Cross-Entropy Loss.
- `weighted_ce`: Weighted Cross-Entropy Loss for handling class imbalance.
- `focal`: Focal Loss for emphasizing hard examples.

Example Usage:
--------------
```python
loss_function = get_loss('focal', gamma=2.0, alpha=0.25)
output = loss_function(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from collections import Counter

def compute_class_weights_from_loader(train_loader, device):
    """
    Extract labels from the DataLoader and compute class weights.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())  # Convert tensor to numpy and store

    return compute_class_weights(all_labels, device)

def compute_class_weights(labels, device):
    """
    Compute class weights based on label distribution.
    """
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Normalize by the maximum weight
    max_weight = max(weights.values())
    normalized_weights = {cls: weight / max_weight for cls, weight in weights.items()}

    return torch.tensor([normalized_weights[i] for i in sorted(normalized_weights.keys())], dtype=torch.float).to(device=device)


# FocalLoss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])  # Alpha for binary case
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))
        pt = (F.softmax(logits, dim=1) * targets_one_hot).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets] if self.alpha.numel() > 1 else \
                      (self.alpha if targets.sum() == 0 else 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        fl = focal_weight * ce_loss

        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        else:
            return fl


# Function to Dynamically Get Loss Function
def get_loss(loss_name, **kwargs):
    """
    Return the appropriate loss function given the loss_name.
    kwargs can contain additional parameters for the loss functions.
    """
    # Define a dictionary mapping loss names to loss constructors
    available_losses = {
        'cross_entropy': nn.CrossEntropyLoss,        # usage: get_loss('cross_entropy')
        'weighted_ce': lambda: nn.CrossEntropyLoss(weight=kwargs.get('weight')),
        'focal': lambda: FocalLoss(gamma=kwargs.get('gamma', 2.0), alpha=kwargs.get('alpha', None))
    }

    if loss_name not in available_losses:
        raise ValueError(f"Unknown loss function '{loss_name}'. Available: {list(available_losses.keys())}")

    # Return the instantiated loss function
    return available_losses[loss_name]()  # Instantiate with the arguments


# Load Loss Function from YAML
def load_loss_from_config(config, device, val_loader=None):
    """
    :param config: configuration from a YAML file
    :return: Instantiated loss function
    """
    if 'loss_name' not in config:
        raise ValueError("YAML file must contain a 'loss_name' field.")

    loss_name = config['loss_name']
    loss_arguments = config.get('loss_arguments') or {}

    if loss_name == "weighted_ce":
        loss_arguments['weight'] = compute_class_weights_from_loader(val_loader, device)
    # Pass the arguments dynamically to get_loss
    return get_loss(loss_name, **loss_arguments)


