import torch


class PreBlock(torch.nn.Module):
    """
    Preprocessing module. It is designed to replace filtering and baseline correction.

    Args:
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
    """
    def __init__(self, input_size=4000):
        super().__init__()
        self.pool1 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool2 = torch.nn.AvgPool1d(kernel_size=13, stride=1, padding=6)
        self.pool3 = torch.nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.ln_1 = torch.nn.LayerNorm(input_size)

    def forward(self, x):
        in_dim = len(x.shape)
        if in_dim>2:
            x = x.squeeze(dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.ln_1(x)
        if in_dim>2:
            x = x.unsqueeze(dim=1)

        return x
    
if __name__=="__main__":
    # Instantiate the model
    input_dim = 2171

    model = PreBlock(input_dim)
    # Create a random test tensor with shape (32, 1, 4000)
    input_tensor = torch.randn(32, 1, input_dim)

    # Forward pass through the model
    output_tensor = model(input_tensor)

    # Print shapes to verify
    print("Input shape:", input_tensor.shape)    # Expected: (32, 1, 4000)
    print("Output shape:", output_tensor.shape)  # Expected: (32, 1, 4000)
    print(len(input_tensor.shape))