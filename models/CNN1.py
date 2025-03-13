import torch
import torch.nn as nn
import torch.nn.functional as F
# from common import PreBlock
from models.common import PreBlock

# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Compress to (batch, channels, 1)
        self.fc1 = nn.Linear(channels, channels // reduction)  # Reduce channels
        self.fc2 = nn.Linear(channels // reduction, channels)  # Expand back
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _ = x.shape
        out = self.global_avg_pool(x).view(batch, channels)  # Global average pooling
        out = F.relu(self.fc1(out))  # FC layer with ReLU
        out = self.fc2(out)  # Expand channels
        out = self.sigmoid(out).view(batch, channels, 1)  # Get channel attention scores
        return x * out  # Scale original input

# Improved Inception Block with SE Block and Residual Connection
class ImprovedInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_dilated, out_1x1pool):
        super(ImprovedInceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)  # 1x1 Conv
        self.branch2 = ConvBlock(in_channels, out_3x3, kernel_size=3, padding=1)  # 3x3 Conv
        self.branch3 = ConvBlock(in_channels, out_dilated, kernel_size=3, dilation=2, padding=2)  # Dilated Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),  # Pooling layer
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)  # 1x1 Conv after pooling
        )
        
        # Total output channels after concatenation
        self.out_channels = out_1x1 + out_3x3 + out_dilated + out_1x1pool
        
        self.se_block = SEBlock(self.out_channels)  # Apply SE Block
        self.residual_conv = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)  # 1x1 Conv to match shapes

    def forward(self, x):
        residual = self.residual_conv(x)  # Match dimensions using 1x1 Conv
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)  # Concatenate features
        # return out + residual  # Apply SE Block and add residual connection
        return self.se_block(out) + residual  # Apply SE Block and add residual connection


# Standard Convolutional Block (Conv -> BatchNorm -> ReLU)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

# Main Model: Inception with SE Blocks and Residuals
class InceptionWithSE(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionWithSE, self).__init__()
        # Replacing kernel_size 10 with kernel_size 5 and dilation=2
        self.conv1 = ConvBlock(in_channels=1, out_channels=16, kernel_size=5, stride=2, dilation=2)  # Initial Conv Layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, dilation=2)  # Second Conv Layer
        
        self.inception1 = ImprovedInceptionBlock(32, 16, 16, 16, 16)  # First Inception Block with SE
        self.inception2 = ImprovedInceptionBlock(64, 32, 32, 32, 32)  # Second Inception Block with SE
        
        self.dropout = nn.Dropout(p=0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(128, 64)  # Fully Connected Layer
        self.fc2 = nn.Linear(64, num_classes)  # Output Layer
        self.bn1 = nn.BatchNorm1d(64)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.global_avg_pool(x).view(batch_size, -1)  # Flatten before FC layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)  # Log Softmax for classification

# Wrapper Model for Pretrained Modules
class ImprovedInception_PreT(nn.Module):
    def __init__(self, num_classes, input_size=4000, pre_module=False):  # Generic number of classes (this can be adjusted)
        super().__init__()
        self.pre_module = pre_module
        if pre_module:
            print("Using Pre module")
        self.pre = PreBlock(input_size=input_size)
        self.IR_PreT = InceptionWithSE(num_classes)

    def forward(self, x):
        if self.pre_module:
            x = self.pre(x)
        x = self.IR_PreT(x)
        return x 


if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(1, 1, 4000).to(device)
    

    model = ImprovedInception_PreT(num_classes=9, input_size=4000, pre_module=False)  # Example for 5-class classification
    model.to(device)
    model.eval()
    total_time = 0.0
    with torch.no_grad():
        for i in range(1000):
            time1 = time.time()
            output = model(input_data)
            time2 = time.time()
            time_per_data = time2-time1
            total_time+=time_per_data
    print(f"time taken is :", total_time)
    print(f"time taken is avarage:", total_time/1000)
    print(f'Inception model output: {output.shape}')
    assert output.shape == (1,9), "Output shape is incorrect."
