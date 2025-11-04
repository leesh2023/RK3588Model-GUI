import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch 
from  torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import math

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)#Fixed random seed

BATCH_SIZE = 10
N_LAYER = 2
N_EPOCHS = 1000
USE_GPU = False
# Data1=pd.read_excel('../1-s2.0-S1359645422007972-mmc70.xlsx')
# Data2=pd.read_excel('../L36_Orthogonal_3Stage_Results.xlsx')
Data1=pd.read_excel('../1-s2.0-S1359645422007972-mmc72.xlsx')
Data2=pd.read_excel('../1-s2.0-S1359645422007972-mmc74.xlsx')
X1_data=Data1.iloc[:,[0,1,2,5,6,7,10,11,12]]
Y1_data=Data1.iloc[:,[3,4,8,9,13,14]]
# X2_data=Data2.iloc[:,[0,1,2,3,4,5,6,7,8]]
# Y2_data=Data2.iloc[:,[5,6,7,8,9,10]]
X2_data=Data2.iloc[:,[0,1,2,5,6,7,10,11,12]]
Y2_data=Data2.iloc[:,[3,4,8,9,13,14]]


X1_data=torch.tensor(np.array(X1_data),dtype=torch.float)
Y1_data=torch.tensor(np.array(Y1_data),dtype=torch.float)
X2_data=torch.tensor(np.array(X2_data),dtype=torch.float)
Y2_data=torch.tensor(np.array(Y2_data),dtype=torch.float)
# 取出25%做测试集，其中测试集1做了*5的数据增强，但是测试集2没做
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1_data,Y1_data,test_size=0.25,random_state=25)
X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2_data,Y2_data,test_size=0.25,random_state=25)
train_dataset1=torch.cat((X1_train,Y1_train),dim=1)#Combined training set
test_dataset1=torch.cat((X1_test,Y1_test),dim=1)#Combined testing set
train_dataset2=torch.cat((X2_train,Y2_train),dim=1)#Combined training set
test_dataset2=torch.cat((X2_test,Y2_test),dim=1)#Combined testing set


train_dataset1=torch.cat([train_dataset1,train_dataset1,train_dataset1,train_dataset1,train_dataset1
                          
                          
                      ],dim=0)
# 注意这里对测试集1重复拼接了5次
test_dataset1=torch.cat([test_dataset1,test_dataset1,test_dataset1,test_dataset1,test_dataset1
                         
                         
                         
                         ],dim=0)
train_dataset=torch.cat([train_dataset1,train_dataset2])



X_train=train_dataset[:,:-6].view((-1,3,3))
Y_train=train_dataset[:,-6:].view((-1,3,2))
Y_shixiaotrain=Y_train[:,2:3,:]
X_test1=test_dataset1[:,:-6].view((-1,3,3))
Y_test1=test_dataset1[:,-6:].view((-1,3,2))
Y_shixiaotest1=Y_test1[:,2:3,:]
X_test2=test_dataset2[:,:-6].view((-1,3,3))
Y_test2=test_dataset2[:,-6:].view((-1,3,2))
Y_shixiaotest2=Y_test2[:,2:3,:]


train_dataset=TensorDataset(X_train,Y_train)
train_length=len(train_dataset)
train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)



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

        self.gru = torch.nn.GRU(input_size, hidden_size, n_layers, bidirectional=bidirectional,batch_first=True)
                                
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def forward(self, input):
        input = input             
        batch_size = input.size(0)
        hidden =self._init_hidden(batch_size)

        output, _ = self.gru(input, hidden)
        fc_output = self.fc(output)
        return fc_output

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return  create_tensor(hidden)


train_loss_min_list=[]
test_loss_min_list1=[]
test_loss_min_list2=[]

train_r_error_dict={}
test1_r_error_dict={}
test2_r_error_dict={}

train_loss_dict={}
test1_loss_dict={}
test2_loss_dict={}

train_loss_min_list_shixiao=[]
test_loss_min_list1_shixiao=[]
test_loss_min_list2_shixiao=[]

train_r_error_dict_shixiao={}
test1_r_error_dict_shixiao={}
test2_r_error_dict_shixiao={}

train_loss_dict_shixiao={}
test1_loss_dict_shixiao={}
test2_loss_dict_shixiao={}
    
for hidden_num in range(2,7,1):
    tr_loss_list=[]
    test_loss_list1=[]
    test_loss_list2=[]
    train_r_error=[]
    test1_r_error=[]
    test2_r_error=[]
    
    tr_loss_list_shixiao=[]
    test_loss_list1_shixiao=[]
    test_loss_list2_shixiao=[]
    train_r_error_shixiao=[]
    test1_r_error_shixiao=[]
    test2_r_error_shixiao=[]
    
    net=RNNClassifier(3,hidden_num,2)
    print(hidden_num)
    criterion=torch.nn.MSELoss(reduction='mean')
    criterion1=torch.nn.MSELoss(reduction='sum')
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)

    for epoch in range(N_EPOCHS):
        e_loss=0
        e_loss1=0
        e_r_error=0
        e_r_error1=0
        for i,(x_input,y_input) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs=net(x_input)
            outputs_shixiao=outputs[:,2:3,:]
            loss=criterion1(outputs,y_input)
            e_loss+=loss
            loss1=criterion1(outputs_shixiao,y_input[:,2:3,:])
            e_loss1+=loss1
            
            b_r_error=(outputs-y_input)/y_input
            b_r_error=torch.abs(b_r_error)
            b_r_error=b_r_error.sum()
            e_r_error=e_r_error+b_r_error
            
            b_r_error1=torch.abs((outputs_shixiao-y_input[:,2:3,:])/y_input[:,2:3,:])
            b_r_error1=b_r_error1.sum()
            e_r_error1=e_r_error1+b_r_error1
            loss.backward()
            optimizer.step()
        
        tr_loss_list_shixiao.append(e_loss1.item()/(Y_shixiaotrain.shape[0]*Y_shixiaotrain.shape[1]*Y_shixiaotrain.shape[2]))
        train_r_error_shixiao.append(e_r_error1.item()/(Y_shixiaotrain.shape[0]*Y_shixiaotrain.shape[1]*Y_shixiaotrain.shape[2]))
        tr_loss_list.append(e_loss.item()/(Y_train.shape[0]*Y_train.shape[1]*Y_train.shape[2]))
        train_r_error.append(e_r_error.item()/(Y_train.shape[0]*Y_train.shape[1]*Y_train.shape[2]))
        with torch.no_grad():
            
            test_pre1=net(X_test1)
            test_loss1_shixiao=criterion(test_pre1[:,2:3,:],Y_shixiaotest1)
            test_loss1=criterion(test_pre1,Y_test1)
            
            
            test_pre2=net(X_test2)
            test_loss2_shixiao=criterion(test_pre2[:,2:3,:],Y_shixiaotest2)
            test_loss2=criterion(test_pre2,Y_test2)
            
                    
            test_loss_list1_shixiao.append(test_loss1_shixiao.item())
            test_loss_list2_shixiao.append(test_loss2_shixiao.item())
            test_loss_list1.append(test_loss1.item())
            test_loss_list2.append(test_loss2.item())
            
            test1_shixiao_r_error=torch.abs((test_pre1[:,2:3,:]-Y_shixiaotest1)/Y_shixiaotest1)
            test1_shixiao_r_error=test1_shixiao_r_error.mean()
            test2_shixiao_r_error=torch.abs((test_pre2[:,2:3,:]-Y_shixiaotest2)/Y_shixiaotest2)
            test2_shixiao_r_error=test2_shixiao_r_error.mean()
            test1_r_error_shixiao.append(test1_shixiao_r_error.item())
            test2_r_error_shixiao.append(test2_shixiao_r_error.item())
            
            test1_r_errors=torch.abs((test_pre1-Y_test1)/Y_test1)
            test1_r_errors=test1_r_errors.mean()
            test2_r_errors=torch.abs((test_pre2-Y_test2)/Y_test2)
            test2_r_errors=test2_r_errors.mean()
            test1_r_error.append(test1_r_errors.item())
            test2_r_error.append(test2_r_errors.item())
      
    train_loss_dict["%s"%hidden_num]=tr_loss_list
    test1_loss_dict["%s"%hidden_num]=test_loss_list1
    test2_loss_dict['%s'%hidden_num]=test_loss_list2
    
    train_r_error_dict['%s'%hidden_num]=train_r_error
    test1_r_error_dict['%s'%hidden_num]=test1_r_error
    test2_r_error_dict['%s'%hidden_num]=test2_r_error
    
    train_loss_dict_shixiao["%s"%hidden_num]=tr_loss_list_shixiao
    test1_loss_dict_shixiao["%s"%hidden_num]=test_loss_list1_shixiao
    test2_loss_dict_shixiao['%s'%hidden_num]=test_loss_list2_shixiao
    
    train_r_error_dict_shixiao['%s'%hidden_num]=train_r_error_shixiao
    test1_r_error_dict_shixiao['%s'%hidden_num]=test1_r_error_shixiao
    test2_r_error_dict_shixiao['%s'%hidden_num]=test2_r_error_shixiao

train_loss=pd.DataFrame(train_loss_dict) # 训练集在每个epoch的损失值
test1_loss=pd.DataFrame(test1_loss_dict) # 测试集1的损失值
test2_loss=pd.DataFrame(test2_loss_dict) # 测试后集2的损失值
train_r_error_data=pd.DataFrame(train_r_error_dict) # 测试集的相对误差
test1_r_error_data=pd.DataFrame(test1_r_error_dict)
test2_r_error_data=pd.DataFrame(test2_r_error_dict)

train_loss_shixiao=pd.DataFrame(train_loss_dict_shixiao)
test1_loss_shixiao=pd.DataFrame(test1_loss_dict_shixiao)
test2_loss_shixiao=pd.DataFrame(test2_loss_dict_shixiao)
train_r_error_data_shixiao=pd.DataFrame(train_r_error_dict_shixiao)
test1_r_error_data_shixiao=pd.DataFrame(test1_r_error_dict_shixiao)
test2_r_error_data_shixiao=pd.DataFrame(test2_r_error_dict_shixiao)

train_loss.to_csv('.\\GRU Orthogonal experimental trainloss augmented.csv')
test1_loss.to_csv('.\\GRU Orthogonal experimental test1loss augmented.csv')
test2_loss.to_csv('.\\GRU Orthogonal experimental test2loss augmented.csv')

train_r_error_data.to_csv('.\\GRU Orthogonal experimental train_r_error augmented.csv')
test1_r_error_data.to_csv('.\\GRU Orthogonal experimental test1_r_error augmented.csv')
test2_r_error_data.to_csv('.\\GRU Orthogonal experimental test2_r_error augmented.csv')

train_loss_shixiao.to_csv('.\\GRU Orthogonal experimental trainloss_aging sample augmented.csv')
test1_loss_shixiao.to_csv('.\\GRU Orthogonal experimental test1loss_aging sample augmented.csv')
test2_loss_shixiao.to_csv('.\\GRU Orthogonal experimental test2loss_aging sample augmented.csv')

train_r_error_data_shixiao.to_csv('.\\GRU Orthogonal experimental train_r_error_aging sample augmented.csv')
test1_r_error_data_shixiao.to_csv('.\\GRU Orthogonal experimental test1_r_error_aging sample augmented.csv')
test2_r_error_data_shixiao.to_csv('.\\GRU Orthogonal experimental test2_r_error_aging sample augmneted.csv')


net.eval()  # 切换到评估模式

# 把 test1 全部样本的预测算出来
with torch.no_grad():
    preds1 = net(X_test1)            # shape [N1, 3, 2]
    preds2 = net(X_test2)            # shape [N2, 3, 2]


# 先把 (255,3,2) reshape 成 (255, 6)
flat1 = preds1.reshape(preds1.shape[0], -1).cpu().numpy()  # shape → (255,6)
flat2 = preds2.reshape(preds2.shape[0], -1).cpu().numpy()  # shape → (255,6)

# 指定 6 列的列名
col_names = [
    'Sol_Yield', 'Sol_Cond',
    'Cold_Yield','Cold_Cond',
    'Aged_Yield','Aged_Cond'
]

df_all1 = pd.DataFrame(flat1, columns=col_names)
df_all2 = pd.DataFrame(flat2, columns=col_names)


# 打印前几个
print("=== Test1 预测前10条（aging 强度，aging 导电率） ===")
print(df_all1.head(10))
print("=== Test2 预测前10条（aging 强度，aging 导电率） ===")
print(df_all2.head(10))

df_all1.to_csv('test1_all_predictions.csv', index=False)
df_all2.to_csv('test2_all_predictions.csv', index=False)

print("已将测试集预测结果保存到 CSV 文件。")
