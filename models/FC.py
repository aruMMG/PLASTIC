import torch.nn as nn
import torch
import torch.nn.functional as F
# from common import PreBlock
from models.common import PreBlock


class FCNet(nn.Module):
    def __init__(self, input_size=4000, num_classes=2, hidden_1=64, hidden_2=128, pre_module=False):
        super().__init__()
        if pre_module:
            print("Pre module is in use")
        self.n_class = num_classes
        self.pre_module = pre_module
        self.pre = PreBlock(input_size=input_size)
        self.fc1 = nn.Linear(input_size,hidden_1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_1, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_2, momentum=0.1)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_2,self.n_class)
    def forward(self, x):
        if self.pre_module:
            x = self.pre(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.out(x)
        return x


    

if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(1, 4000).to(device)
    model = FCNet(num_classes=9,pre_module=False)

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