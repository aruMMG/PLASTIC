"""
Optimizer Documentation
------------------------

Optimizers determine how the model learns from data by updating the parameters 
based on the gradients of the loss function with respect to the model parameters.
This module provides a dynamic optimizer factory for creating various optimizers, 
along with background theory, mathematical formulations, and practical usage 
guidance to help users make informed choices.

---

Gradient Descent Optimizers
---------------------------

Gradient Descent forms the foundation of most optimizers. It updates model parameters 
`theta` by moving in the direction opposite to the gradient of the loss function:

    theta = theta - alpha * grad(L(theta))

Where:
- `alpha`: Learning rate, which determines the step size.

Modern optimizers enhance this approach to improve convergence, stability, and adaptability.

---

Implemented Optimizers
----------------------

1. **Adam (Adaptive Moment Estimation)**

   Description:
   - Combines momentum and adaptive learning rates.
   - Maintains moving averages of gradients (first moment) and squared gradients (second moment).

   Mathematics:
       m_t = beta1 * m_{t-1} + (1 - beta1) * grad
       v_t = beta2 * v_{t-1} + (1 - beta2) * (grad^2)
       m_hat = m_t / (1 - beta1^t)
       v_hat = v_t / (1 - beta2^t)
       theta = theta - alpha * (m_hat / (sqrt(v_hat) + eps))

   Use Cases:
   - General-purpose optimizer.
   - Effective for deep networks and large datasets.

---

2. **AdamW (Adam with Weight Decay)**

   Description:
   - Decouples weight decay from gradient updates for better regularization.

   Use Cases:
   - Suitable for models requiring weight decay regularization, such as transformers.

---

3. **SGD (Stochastic Gradient Descent)**

   Description:
   - Updates parameters using mini-batches of data.
   - Adding momentum improves convergence and dampens oscillations.

   Mathematics:
       v_t = beta * v_{t-1} + alpha * grad
       theta = theta - v_t

   Use Cases:
   - Preferred for tasks like image classification when combined with learning rate schedules.

---

4. **RAdam (Rectified Adam)**

   Description:
   - Adds a rectification term to adaptively adjust learning rates during early training stages.

   Mathematics:
       r_t = min(1, (sqrt(t) - beta) / (sqrt(t) + beta))
       theta = theta - alpha * r_t * (m_hat / (sqrt(v_hat) + eps))

   Use Cases:
   - Scenarios requiring stable training and convergence.

---

5. **Adagrad**

   Description:
   - Adapts learning rates for each parameter based on the history of gradients.

   Mathematics:
       theta = theta - alpha * grad / (sqrt(sum(grad^2)) + eps)

   Use Cases:
   - Problems with sparse features (e.g., NLP, recommendation systems).

---

6. **RMSProp**

   Description:
   - Introduces a moving average of squared gradients to prevent rapid learning rate decay.

   Mathematics:
       v_t = beta * v_{t-1} + (1 - beta) * (grad^2)
       theta = theta - alpha * grad / (sqrt(v_t) + eps)

   Use Cases:
   - Recurrent neural networks and tasks requiring adaptive learning rates.

---

7. **Lookahead Optimizer**

   Description:
   - Enhances base optimizers (e.g., Adam, SGD) by maintaining "fast weights" and "slow weights."

   Mathematics:
       theta_slow = theta_slow + alpha * (theta_fast - theta_slow)

   Use Cases:
   - Improves performance and reduces sensitivity to learning rate.

---

Dynamic Optimizer Selection
---------------------------

The `OptimizerFactory` class enables dynamic creation of optimizers. Supported optimizers:
- `adam`
- `adamw`
- `sgd`
- `radam`
- `adagrad`
- `rmsprop`
- `lookahead_adam`
- `lookahead_sgd`

Example Usage:
--------------
```python
factory = OptimizerFactory(model.parameters())
optimizer = factory.get_optimizer('adam', lr=0.001, weight_decay=0.01)
"""

import torch
import torch.optim as optim
import inspect

try:
    import torch_optimizer as extra_optim
except ImportError:
    extra_optim = None

class OptimizerFactory:
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters

    def _get_default_args(self, optimizer_class):
        """
        Retrieves the default arguments of an optimizer class.
        """
        signature = inspect.signature(optimizer_class)
        return {param.name: param.default for param in signature.parameters.values() if param.default is not param.empty}

    def _merge_with_defaults(self, optimizer_class, kwargs):
        """
        Merges user-provided arguments with the default arguments of the optimizer.
        """
        defaults = self._get_default_args(optimizer_class)
        return {**defaults, **kwargs}

    def create_adam(self, **kwargs):
        merged_kwargs = self._merge_with_defaults(optim.Adam, kwargs)
        return optim.Adam(self.model_parameters, **merged_kwargs)
    
    # def create_adam(self, **kwargs):
    #     return optim.Adam(
    #         self.model_parameters,
    #         lr=kwargs.get('lr', 1e-3),
    #         betas=tuple(kwargs.get('betas', [0.9, 0.999])),
    #         eps=float(kwargs.get('eps', 1e-8)),
    #         weight_decay=kwargs.get('weight_decay', 0.0)
    #     )

    def create_adamw(self, **kwargs):

        return optim.AdamW(
            self.model_parameters,
            lr=kwargs.get('lr', 1e-3),
            betas=tuple(kwargs.get('betas', [0.9, 0.999])),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    
    def create_sgd(self, **kwargs):
        return optim.SGD(
            self.model_parameters,
            lr=kwargs.get('lr', 0.01),
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )

    def create_radam(self, **kwargs):
        if extra_optim is None:
            raise ImportError("RAdam requires `torch_optimizer`. Install with `pip install torch_optimizer`.")
        return extra_optim.RAdam(
            self.model_parameters,
            lr=kwargs.get('lr', 1e-3),
            betas=tuple(kwargs.get('betas', [0.9, 0.999])),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )

    def create_adagrad(self, **kwargs):
        return optim.Adagrad(
            self.model_parameters,
            lr=kwargs.get('lr', 1e-2),
            lr_decay=kwargs.get('lr_decay', 0.0),
            weight_decay=kwargs.get('weight_decay', 0.0),
            initial_accumulator_value=kwargs.get('initial_accumulator_value', 0.0),
            eps=kwargs.get('eps', 1e-10)
        )

    def create_rmsprop(self, **kwargs):
        return optim.RMSprop(
            self.model_parameters,
            lr=kwargs.get('lr', 1e-2),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0),
            momentum=kwargs.get('momentum', 0.0),
            centered=kwargs.get('centered', False)
        )

    def create_lookahead_adam(self, **kwargs):
        if extra_optim is None:
            raise ImportError("Lookahead requires `torch_optimizer`. Install with `pip install torch_optimizer`.")
        base_optimizer = optim.Adam(
            self.model_parameters,
            lr=kwargs.get('lr', 1e-3),
            betas=tuple(kwargs.get('betas', [0.9, 0.999])),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
        return extra_optim.Lookahead(
            base_optimizer,
            alpha=kwargs.get('alpha', 0.5),
            k=kwargs.get('k', 5)
        )

    def create_lookahead_sgd(self, **kwargs):
        if extra_optim is None:
            raise ImportError("Lookahead requires `torch_optimizer`. Install with `pip install torch_optimizer`.")
        base_optimizer = optim.SGD(
            self.model_parameters,
            lr=kwargs.get('lr', 0.01),
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
        return extra_optim.Lookahead(
            base_optimizer,
            alpha=kwargs.get('alpha', 0.5),
            k=kwargs.get('k', 5)
        )

    def get_optimizer(self, optimizer_name, **kwargs):
        # Map optimizer names to creation methods
        method_map = {
            'adam': self.create_adam,
            'adamw': self.create_adamw,
            'sgd': self.create_sgd,
            'radam': self.create_radam,
            'adagrad': self.create_adagrad,
            'rmsprop': self.create_rmsprop,
            'lookahead_adam': self.create_lookahead_adam,
            'lookahead_sgd': self.create_lookahead_sgd
        }

        if optimizer_name not in method_map:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available: {list(method_map.keys())}")

        return method_map[optimizer_name](**kwargs)

def load_optimizer_from_config(config, model_parameters):
    """
    :param config: configuration from a YAML file (dict)
    :param model_parameters: the parameters of the model to be optimized (e.g., model.parameters())
    :return: Instantiated optimizer
    """
    if 'optimizer_name' not in config:
        raise ValueError("YAML file must contain an 'optimizer_name' field.")

    optimizer_name = config['optimizer_name']
    optimizer_arguments = config.get('optimizer_arguments', {})
    print(optimizer_arguments)

    factory = OptimizerFactory(model_parameters)
    optimizer = factory.get_optimizer(optimizer_name, **optimizer_arguments)
    return optimizer

# Example usage:
# config = {
#     'optimizer_name': 'radam',
#     'optimizer_arguments': {
#         'lr': 0.001,
#         'weight_decay': 0.0,
#         'betas': [0.9, 0.999]
#     }
# }
# model = SomeModel()
# optimizer = load_optimizer_from_config(config, model.parameters())
