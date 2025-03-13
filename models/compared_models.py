import torch.nn as nn
import torch
import torch.nn.functional as F


class NNH10(nn.Module):
    def __init__(self, input_size=4000, num_classes=2, hidden_1=10):
        super().__init__()
        """
            Hyperspectral imaging-based classification of post-consumer
        thermoplastics for plastics recycling using artificial neural network 
        """

        self.n_class = num_classes
        self.fc1 = nn.Linear(input_size,hidden_1)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(hidden_1,self.n_class)
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.out(x)
        return F.softmax(x)

class NNHout(nn.Module):
    def __init__(self, input_size=4000, num_classes=2):
        super().__init__()
        """
            Hyperspectral imaging-based classification of post-consumer
        thermoplastics for plastics recycling using artificial neural network 
        """

        self.n_class = num_classes
        self.fc1 = nn.Linear(input_size,self.n_class)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.n_class,self.n_class)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        return F.softmax(x)
    
class CNN(nn.Module):
    def __init__(self, num_classes=2, input_size=4000, in_channels=1):
        super().__init__()
        """
        Convolutional neural network for simultaneous prediction of several soil properties using visible/near-infrared, mid-infrared, and their combined spectra
        Detection of Plastic Granules and Their Mixtures
        """
        self.n_class = num_classes
        self.in_channels = in_channels
        self.input_size = input_size
        self.conv1 = nn.Sequential()
        self.conv1.add_module("Conv1", nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, padding=3))
        self.conv1.add_module("relu1", nn.ReLU())
        self.conv1.add_module("maxpool1", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential()
        self.conv2.add_module("Conv2", nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, padding=3))
        self.conv2.add_module("relu2", nn.ReLU())
        self.conv2.add_module("maxpool2", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential()
        self.conv3.add_module("Conv3", nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, padding=3))
        self.conv3.add_module("relu3", nn.ReLU())
        self.conv3.add_module("maxpool3", nn.MaxPool1d(kernel_size=5, stride=5))
        self.conv4 = nn.Sequential()
        self.conv4.add_module("Conv4", nn.Conv1d(in_channels=128, out_channels=256, kernel_size=20, padding=3))
        self.conv4.add_module("relu4", nn.ReLU())
        self.conv4.add_module("maxpool4", nn.MaxPool1d(kernel_size=5, stride=5))
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc1 = None
        # self.fc1 = nn.Linear(256*40,128)
        self.out = nn.Linear(128,self.n_class)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout1(x)
        if self.fc1 is None:
            fc_input_size = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(fc_input_size, 128).to(x.device)
        # print(x.shape)
        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.out(x)
        x = F.softmax(x,dim=1)
        return x
    def initialize_fc1(self, x):
        """Manually initialize fc1 before loading state_dict"""
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
        x_shape = x.shape
        self.fc1_input_size = x_shape[1] * x_shape[2]
        self.fc1 = nn.Linear(self.fc1_input_size, 128)  
if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_data = torch.randn(1, 1, 4000).to(device)
    # model = CNN(9)
    input_data = torch.randn(1, 4000).to(device)
    # model = NNH10(num_classes=9)
    model = NNHout(num_classes=9)
    model.to(device)
    model.eval()
    total_time = 0.0
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