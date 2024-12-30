import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data as gData
from torch_geometric.loader import DataLoader as gDataLoader
from Simulation import Simulation_OTU_Robust

   
class AE_DataSet(Dataset):
    def __init__(self,pair_set):
        super(AE_DataSet).__init__()
        self.pair_set = pair_set
        self.data_list=[]
        for index in range(len(self.pair_set)):
            tmp=[]
            x = torch.from_numpy(np.vstack((self.pair_set[index]['real'],self.pair_set[index]['sim'])))
            tmp.append(x)
            tmp.append(torch.tensor(self.pair_set[index]['Wii']))
            tmp.append(torch.tensor(self.pair_set[index]['Wij']))
            tmp.append(torch.tensor(self.pair_set[index]['growth_rate']))
            self.data_list.append(tmp)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        feature = self.data_list[index][0].float()
        Wii_lable = self.data_list[index][-3].float()
        Wij_lable = self.data_list[index][-2].float()
        Ui_lable = self.data_list[index][-1].float()
        return feature,Wii_lable,Wij_lable,Ui_lable
    
class Ne_AE_DataSet(Dataset):
    def __init__(self,pair_set):
        super(Ne_AE_DataSet).__init__()
        self.pair_set = pair_set
        self.data_list=[]
        for index in range(len(self.pair_set)):
            tmp=[]
            x = torch.from_numpy(np.vstack((self.pair_set[index]['no_real'],self.pair_set[index]['sim'])))
            tmp.append(x)
            tmp.append(torch.tensor(self.pair_set[index]['Wii']))
            tmp.append(torch.tensor(0)) #wij
            tmp.append(torch.tensor(self.pair_set[index]['growth_rate']))
            self.data_list.append(tmp)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        feature = self.data_list[index][0].float()
        Wii_lable = self.data_list[index][-3].float()
        Wij_lable = self.data_list[index][-2].float()
        Ui_lable = self.data_list[index][-1].float()
        return feature,Wii_lable,Wij_lable,Ui_lable
    
class AE_3_DataSet(Dataset):
    def __init__(self,pair_set):
        super(AE_3_DataSet).__init__()
        self.pair_set = pair_set
        self.data_list=[]
        for index in range(len(self.pair_set)):
            tmp=[]
            x = torch.from_numpy(np.vstack((self.pair_set[index]['real'],self.pair_set[index]['sim'])))
            tmp.append(x)
            tmp.append(torch.tensor(self.pair_set[index]['Wii']))
            tmp.append(torch.tensor(self.pair_set[index]['Wij']))
            tmp.append(torch.tensor(self.pair_set[index]['growth_rate']))
            self.data_list.append(tmp)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        feature = self.data_list[index][0].float()
        Wii_lable = self.data_list[index][-3].float()
        Wij_lable = self.data_list[index][-2].float()
        Ui_lable = self.data_list[index][-1].float()
        lable = torch.empty(size=(1, 3))
        lable[:,0] = Ui_lable
        lable[:,1] = Wii_lable
        lable[:,2] = Wij_lable
        return feature, lable
    
class Ne_AE_3_DataSet(Dataset):
    def __init__(self,pair_set):
        super(Ne_AE_3_DataSet).__init__()
        self.pair_set = pair_set
        self.data_list=[]
        for index in range(len(self.pair_set)):
            tmp=[]
            x = torch.from_numpy(np.vstack((self.pair_set[index]['no_real'],self.pair_set[index]['sim'])))
            tmp.append(x)
            tmp.append(torch.tensor(self.pair_set[index]['Wii']))
            tmp.append(torch.tensor(0)) #wij
            tmp.append(torch.tensor(self.pair_set[index]['growth_rate']))
            self.data_list.append(tmp)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        feature = self.data_list[index][0].float()
        Wii_lable = self.data_list[index][-3].float()
        Wij_lable = self.data_list[index][-2].float()
        Ui_lable = self.data_list[index][-1].float()
        lable = torch.empty(size=(1, 3))
        lable[:,0] = Ui_lable
        lable[:,1] = Wii_lable
        lable[:,2] = Wij_lable
        return feature, lable

#-------------------------------------------------------------------------------------

class MLP_DataSet(Dataset):
    def __init__(self,pair_set):
        super(MLP_DataSet).__init__()
        self.pair_set = pair_set
        self.data_list=[]
        for index in range(len(self.pair_set)):
            tmp=[]
            x = torch.from_numpy(np.vstack((self.pair_set[index]['real'],self.pair_set[index]['sim'])))
            tmp.append(x)
            tmp.append(torch.tensor(self.pair_set[index]['Wij']))
            tmp.append(torch.tensor(self.pair_set[index]['growth_rate']))
            self.data_list.append(tmp)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        feature = self.data_list[index][0].float()
        Wij_lable = self.data_list[index][-2].float()
        Ui_lable = self.data_list[index][-1].float()
        return feature,Wij_lable,Ui_lable
    
class CNN_DataSet(Dataset):
    def __init__(self,pair_set):
        super(CNN_DataSet).__init__()
        self.pair_set = pair_set
        self.data_list=[]
        for index in range(len(self.pair_set)):
            tmp=[]
            x = torch.from_numpy(np.vstack((self.pair_set[index]['real'],self.pair_set[index]['sim'])))
            tmp.append(x)
            tmp.append(torch.tensor(self.pair_set[index]['Wij']))
            tmp.append(torch.tensor(self.pair_set[index]['growth_rate']))
            self.data_list.append(tmp)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        feature = self.data_list[index][0].float()
        Wij_lable = self.data_list[index][-2].float()
        Ui_lable = self.data_list[index][-1].float()
        return feature,Wij_lable,Ui_lable
    
class GNN_DataLoader():
    def __init__(self,np_data,t_index,sim_nums,batch_size,split_rate):
        """
        :param np_data:OTU的时间序列矩阵
        :param t_index:每一个时间点的index
        :param sim_nums:simulation的OTU的数量
        :param batch_size:batch_size的大小
        :param split_rate:训练集与测试集的分割比例
        """
        self.pair_set = Simulation_OTU_Robust(np_data,sim_nums,t_index)
        self.data_list=[] 
        for index in range(len(self.pair_set)):
            x = torch.from_numpy(np.vstack((self.pair_set[index]['real'],self.pair_set[index]['sim'])))
            datax = gData(x=x,edge_index=torch.tensor([[0],[1]], dtype=torch.long),y=torch.tensor([self.pair_set[index]['Wij']]))
            self.data_list.append(datax)
        random.shuffle(self.data_list)
        self.train_data_list = self.data_list[:int(len(self.data_list)*split_rate)]
        self.test_data_list = self.data_list[int(len(self.data_list)*split_rate):]
        self.train_loader = gDataLoader(self.train_data_list, batch_size=batch_size)
        self.test_loader = gDataLoader(self.test_data_list, batch_size=batch_size)