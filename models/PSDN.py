import torch.nn as nn
import torch
import torch.nn.functional as F

# from common import PreBlock
from models.common import PreBlock

class ResidualNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.n_class = num_classes
        self.in_channels = in_channels
        self.conv1 = nn.Sequential()
        self.conv1.add_module("Conv1", nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3))
        self.conv1.add_module("relu1", nn.ReLU())
        self.conv1.add_module("maxpool1", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential()
        self.conv2.add_module("Conv2", nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3))
        self.conv2.add_module("relu2", nn.ReLU())
        self.conv2.add_module("maxpool2", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential()
        self.conv3.add_module("Conv3", nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3))
        self.conv3.add_module("relu3", nn.ReLU())
        self.conv3.add_module("maxpool3", nn.MaxPool1d(kernel_size=5, stride=5))
        self.conv4 = nn.Sequential()
        self.conv4.add_module("Conv4", nn.Conv1d(in_channels=148, out_channels=256, kernel_size=7, padding=3))
        self.conv4.add_module("relu4", nn.ReLU())
        self.conv4.add_module("maxpool4", nn.MaxPool1d(kernel_size=5, stride=5))

        self.fc1 = None
        # self.fc1 = nn.Linear(256*40,128)
        self.out = nn.Linear(128,self.n_class)
    def forward(self, x):
        batch_size = x.size(0)
        x1 = x
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x1 = x1.reshape(x.shape[0],-1,x.shape[2])
        x = torch.cat((x,x1),dim=1)
        x = self.conv4(x)

        if self.fc1 is None:
            fc_input_size = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(fc_input_size, 128).to(x.device)
        # print(x.shape)
        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x,dim=1)
        return x


class InceptionNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionNetwork, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_block(in_channels=1, out_channels=8, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(8, 16, 10, 2)
        self.inception1 = Naive_inception_block(16, 8, 8, 8, 8)
        # self.fc1 = nn.Linear(247*32, 128)
        self.fc1_input_size = None
        self.fc1 = None
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.inception1(x)

        if self.fc1 is None:
            self.fc_input_size = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(self.fc_input_size, 128).to(x.device)

        # x = x.view(-1, 247*32)
        x = x.view(batch_size, -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)
    def initialize_fc1(self, x_shape):
        """Manually initialize fc1 before loading state_dict"""
        self.fc1_input_size = x_shape[1] * x_shape[2]
        self.fc1 = nn.Linear(self.fc1_input_size, 128)  

        
class InceptionNetwork_PreT(nn.Module):
    def __init__(self, num_classes, input_size=4000, pre_module=False): # generic number of classes (this can be adjusted)
        super().__init__()
        self.pre_module = pre_module
        if pre_module:
            print("Using Pre module")
        self.pre = PreBlock(input_size=input_size)
        self.IR_PreT = InceptionNetwork(num_classes)


    def forward(self, x):
        if self.pre_module:
            x = self.pre(x)
        x = self.IR_PreT(x)
        return x 
    def initialize_fc1(self, dummy_input):
        """Perform a partial forward pass to compute fc1 input size"""
        with torch.no_grad():
            if self.pre_module:
                dummy_input = self.pre(dummy_input)  # Process only if pre-module is used
            dummy_input = self.IR_PreT.conv1(dummy_input)
            dummy_input = F.max_pool1d(dummy_input, 2)
            dummy_input = F.relu(self.IR_PreT.conv2(dummy_input))
            dummy_input = F.max_pool1d(dummy_input, 2)
            dummy_input = self.IR_PreT.inception1(dummy_input)

            # Now we have the correct shape for fc1
            self.IR_PreT.initialize_fc1(dummy_input.shape)

class Naive_inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_1x1pool):
        super(Naive_inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = conv_block(in_channels, out_3x3, kernel_size=3, padding=1)
        self.branch3 = conv_block(in_channels, out_5x5, kernel_size=5, padding=2)     
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )
        
    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.conv(x))
    



if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(1, 1, 4000).to(device)
    

    model = InceptionNetwork(9)
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

    # model = ResidualNet()
    # model.to(device)
    # model.eval()
    # output = model(input_data)
    # print(f'ResidualNet model output: {output.shape}')
    # assert output.shape == (32,2), "Output shape is incorrect."
