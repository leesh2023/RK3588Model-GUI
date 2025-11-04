import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import math
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

BATCH_SIZE = 10
N_LAYER = 2
N_EPOCHS = 1000
USE_GPU = False

# 数据加载
Data1=pd.read_excel('../1-s2.0-S1359645422007972-mmc70.xlsx')
Data2=pd.read_excel('../1-s2.0-S1359645422007972-mmc66.xlsx')

X1_data = Data1.iloc[:, [0,1,2,5,6,7,10,11,12]]
Y1_data = Data1.iloc[:, [3,4,8,9,13,14]]
X2_data = Data2.iloc[:, [0,1,2,5,6,7,10,11,12]]
Y2_data = Data2.iloc[:, [3,4,8,9,13,14]]

X1_data = torch.tensor(np.array(X1_data), dtype=torch.float)
Y1_data = torch.tensor(np.array(Y1_data), dtype=torch.float)
X2_data = torch.tensor(np.array(X2_data), dtype=torch.float)
Y2_data = torch.tensor(np.array(Y2_data), dtype=torch.float)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1_data, Y1_data, test_size=0.25, random_state=25)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2_data, Y2_data, test_size=0.25, random_state=25)

train_dataset1 = torch.cat((X1_train, Y1_train), dim=1)
test_dataset1 = torch.cat((X1_test, Y1_test), dim=1)
train_dataset2 = torch.cat((X2_train, Y2_train), dim=1)
test_dataset2 = torch.cat((X2_test, Y2_test), dim=1)

train_dataset1 = torch.cat([train_dataset1]*5, dim=0)
test_dataset1 = torch.cat([test_dataset1]*5, dim=0)
train_dataset = torch.cat([train_dataset1, train_dataset2])

X_train = train_dataset[:, :-6].view((-1, 3, 3))
Y_train = train_dataset[:, -6:].view((-1, 3, 2))
Y_shixiaotrain = Y_train[:, 2:3, :]

X_test1 = test_dataset1[:, :-6].view((-1, 3, 3))
Y_test1 = test_dataset1[:, -6:].view((-1, 3, 2))
Y_shixiaotest1 = Y_test1[:, 2:3, :]

X_test2 = test_dataset2[:, :-6].view((-1, 3, 3))
Y_test2 = test_dataset2[:, -6:].view((-1, 3, 2))
Y_shixiaotest2 = Y_test2[:, 2:3, :]

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.gru = torch.nn.GRU(input_size, hidden_size, n_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self._init_hidden(batch_size)
        output, _ = self.gru(input, hidden)
        fc_output = self.fc(output)
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

# 保存路径
os.makedirs("onnx_models", exist_ok=True)

# for hidden_num in range(2, 9, 1):
hidden_num = 8
net = RNNClassifier(3, hidden_num, 2)
print(f"Training GRU with hidden_size = {hidden_num}")
criterion = torch.nn.MSELoss(reduction='mean')
criterion1 = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(N_EPOCHS):
    net.train()
    for x_input, y_input in train_loader:
        optimizer.zero_grad()
        outputs = net(x_input)
        outputs_shixiao = outputs[:, 2:3, :]
        loss = criterion1(outputs, y_input)
        loss.backward()
        optimizer.step()

# 导出 ONNX 模型
net.eval()
dummy_input = torch.randn(1, 3, 3)
if USE_GPU:
    net = net.cuda()
    dummy_input = dummy_input.cuda()
onnx_path = f"onnx_models/gru_model_hidden{hidden_num}_version3.onnx"
torch.onnx.export(
    net,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"已保存 ONNX 模型: {onnx_path}")
