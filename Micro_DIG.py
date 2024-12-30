from Simulation import Simulation_X
from Utils import adjust_concentrations,fix_random_seeds
from Network import AE_3_model
from Data_process import AE_3_DataSet,Ne_AE_3_DataSet
from Train_process import AE3_train

from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import pickle as pkl


class MicroDIG:
    def __init__(self,input_data,input_t_index,isNormal,result_save_name):
        fix_random_seeds(42)
        self.raw_data = input_data
        self.t_index = input_t_index
        if isNormal:
            self.raw_data = adjust_concentrations(self.raw_data)

        self.result_save_name = os.path.dirname(__file__) + "/" + result_save_name

        if not os.path.exists(self.result_save_name):
            os.mkdir(self.result_save_name)


    def Generate_Enhanced_data(self):
        self.enhanced_data = Simulation_X(self.raw_data,self.t_index,6000)

        
    def Get_trainAndval_set(self): 
        self.train_set, self.val_set = train_test_split(self.enhanced_data, test_size=0.2, random_state=42)
        self.train_DataSet = AE_3_DataSet(self.train_set)
        self.val_DataSet = AE_3_DataSet(self.val_set)
        self.train_iter=DataLoader(self.train_DataSet,batch_size=32,shuffle=True)
        self.val_iter=DataLoader(self.val_DataSet,batch_size=32,shuffle=True)  
        
    
    def train_model(self,device,loss_log_name):
        self.model = AE_3_model(self.raw_data.shape[1])
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        AE3_train(self.model,self.device,self.train_iter,self.val_iter,self.result_save_name,loss_log_name)


    def load_model(self,model_path):
        self.model = torch.load(model_path)
        self.device = next(self.model.parameters()).device

    
    def get_Wij_without_check(self):
        Wij_without_check = np.zeros(shape=(self.raw_data.shape[0],self.raw_data.shape[0]))
        for iq in tqdm(range(self.raw_data.shape[0]),position=0,desc="i",colour='green',ncols=80):
            for jq in tqdm(range(self.raw_data.shape[0]),position=1,desc="j",leave=False,colour='red',ncols=80):
                corr_output = self.model(torch.from_numpy(np.vstack((self.raw_data[jq],self.raw_data[iq]))).float().to("cuda:0"))[1].to("cpu").squeeze().detach().numpy()
                Wij_without_check[iq][jq] = corr_output[-1]
        return Wij_without_check


    def glvModel(t,x0,growth_rate,corr_matrix):
        dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
        return dx


    def Get_Two_Distributions(self):
        train_DataSet_small,_ = torch.utils.data.random_split(self.train_DataSet,[int(len(self.train_DataSet)*0.25),int(len(self.train_DataSet)*0.75)])
        train_po_simdata = []
        pvalue_iter1=DataLoader(train_DataSet_small,batch_size=1,shuffle=False)
        for k,(pvalue_data,label) in enumerate(tqdm(pvalue_iter1,position=0,desc="po_distribution_generating_1",colour='red',ncols=80)):
            pvalue_data = pvalue_data.to(self.device)
            _,corr_output = self.model(pvalue_data)
            corr_output = corr_output.to("cpu").squeeze().detach().numpy()
            label = label.squeeze().numpy()
            Ui_pre = copy.deepcopy(corr_output[0])
            Wii_pre = copy.deepcopy(corr_output[1])
            Wij_pre = copy.deepcopy(corr_output[-1])
            pvalue_data = pvalue_data.to("cpu").squeeze().numpy()
            pvalue_real = copy.deepcopy(pvalue_data[0,:])
            pvalue_sim = copy.deepcopy(pvalue_data[1,:])
            
            sim_array = np.zeros(shape=(10,pvalue_real.shape[0]))
            
            get_set = True
            try:
                for epoch in range(10):
                    Ui = Ui_pre + np.random.normal(0,0.01)
                    Wij = copy.deepcopy(Wij_pre)
                    #构造相关性矩阵
                    corr_matrix = np.zeros(shape=(2,2))
                    row, col = np.diag_indices_from(corr_matrix)                   
                    corr_matrix[row,col] = np.array([Wii_pre+np.random.normal(0,0.01) for i in range(2)])
                    corr_matrix[1,0]=copy.deepcopy(Wij)
                    #构造自增长率
                    growth_rate = np.array([Ui,Ui])
                    #构造初值
                    otu_0 = np.zeros(shape=2)
                    otu_0[0] = copy.deepcopy(pvalue_real[0])
                    otu_0[1] = np.mean(pvalue_real)

                    # 对输入的拼接数据进行切段
                    head_index = 0
                    cut_set = []
                    for i3 in range(self.t_index.shape[0]):
                        if (self.t_index[i3] < self.t_index[i3-1]) and (i3 != 0):
                            tmp1 = {}
                            tmp1["cut_data"] = copy.deepcopy(pvalue_real[head_index:i3])
                            tmp1["cut_time"] = copy.deepcopy(self.t_index[head_index:i3])
                            cut_set.append(tmp1)
                            head_index = copy.deepcopy(i3)
                        if i3 == pvalue_real.shape[0]-1:
                            tmp1 = {}
                            tmp1["cut_data"] = copy.deepcopy(pvalue_real[head_index:])
                            tmp1["cut_time"] = copy.deepcopy(self.t_index[head_index:])
                            cut_set.append(tmp1)
                    # 进行glv方程模拟
                    sim_set = []
                    for i4 in range(len(cut_set)):
                        cut_data = copy.deepcopy(cut_set[i4]["cut_data"])
                        cut_time = copy.deepcopy(cut_set[i4]["cut_time"])
                        t_step = cut_time.shape[0]
                        otu_0[0] = copy.deepcopy(cut_data[0])
                        otu_0[1] = np.mean(cut_data)
                        sim_otu = np.zeros(shape=(1,t_step))
                        sim_otu[:,0] = copy.deepcopy(otu_0[1:])

                        for i5 in range(t_step-1):
                            sol = solve_ivp(self.glvModel,[0,cut_time[i5+1]-cut_time[i5]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,cut_time[i5+1]-cut_time[i5]])
                            sim_otu[:,i5+1] = copy.deepcopy(sol.y[:,1][1:])
                            otu_0[:1] = copy.deepcopy(cut_data[i5+1])
                            otu_0[1:] = copy.deepcopy(sol.y[:,1][1:])
                        sim_set.append(sim_otu)
                    
                    finanl_sim = np.zeros(shape=(1,self.t_index.shape[0]))
                    head_index = 0
                    for i6 in range(len(sim_set)):
                        gap = sim_set[i6].shape[1]
                        finanl_sim[:,head_index:head_index+gap] = copy.deepcopy(sim_set[i6])
                        head_index += gap

                    sim_array[epoch,:] = copy.deepcopy(finanl_sim)
            except:
                get_set = False
            if get_set:
                tmp = {}
                tmp["original_real"] = pvalue_real
                tmp["original_sim"] = pvalue_sim
                tmp["sim_datas"] = sim_array
                tmp["Ui"] = Ui_pre
                tmp["Wii"] = Wii_pre
                tmp["Wij"] = Wij_pre
                train_po_simdata.append(tmp)

        
        ne_DataSet = Ne_AE_3_DataSet(self.train_set)
        ne_wij = []
        pvalue_iter2=DataLoader(ne_DataSet,batch_size=1,shuffle=False)
        for k,(pvalue_data,label) in enumerate(pvalue_iter2):
            pvalue_data = pvalue_data.to(self.device)
            _,corr_output = self.model(pvalue_data)
            corr_output = corr_output.to("cpu").squeeze().detach().numpy()
            Ui_pre = corr_output[0]
            Wii_pre = corr_output[1]
            Wij_pre = corr_output[-1]
            ne_wij.append(Wij_pre)
        ne_wij_min,ne_wij_max = np.percentile(ne_wij, [25,75])

        ne_DataSet_small,_ = torch.utils.data.random_split(ne_DataSet,[int(len(ne_DataSet)*0.25),int(len(ne_DataSet)*0.75)])
        train_ne_simdata = []
        pvalue_iter2=DataLoader(ne_DataSet_small,batch_size=1,shuffle=False)
        for k,(pvalue_data,label) in enumerate(tqdm(pvalue_iter2,position=0,desc="ne_distribution_generating_1",colour='green',ncols=80)):
            pvalue_data = pvalue_data.to(self.device)
            _,corr_output = self.model(pvalue_data)
            corr_output = corr_output.to("cpu").squeeze().detach().numpy()
            Ui_pre = copy.deepcopy(corr_output[0])
            Wii_pre = copy.deepcopy(corr_output[1])
            Wij_pre = copy.deepcopy(corr_output[-1])
            if (Wij_pre > ne_wij_min) & (Wij_pre < ne_wij_max):
                continue
            pvalue_data = pvalue_data.to("cpu").squeeze().numpy()
            pvalue_real = copy.deepcopy(pvalue_data[0,:])
            pvalue_sim = copy.deepcopy(pvalue_data[1,:])
            
            sim_array = np.zeros(shape=(10,pvalue_real.shape[0]))
            
            get_set = True
            try:
                for epoch in range(10):
                    Ui = Ui_pre + np.random.normal(0,0.01)
                    Wij = copy.deepcopy(Wij_pre)
                    #构造相关性矩阵
                    corr_matrix = np.zeros(shape=(2,2))
                    row, col = np.diag_indices_from(corr_matrix)                   
                    corr_matrix[row,col] = np.array([Wii_pre+np.random.normal(0,0.01) for i in range(2)])
                    corr_matrix[1,0]=copy.deepcopy(Wij)
                    #构造自增长率
                    growth_rate =  np.array([Ui,Ui])
                    #构造初值
                    otu_0 = np.zeros(shape=2)
                    otu_0[0] = copy.deepcopy(pvalue_real[0])
                    otu_0[1] = np.mean(pvalue_real)

                    # 对输入的拼接数据进行切段
                    head_index = 0
                    cut_set = []
                    for i3 in range(self.t_index.shape[0]):
                        if (self.t_index[i3] < self.t_index[i3-1]) and (i3 != 0):
                            tmp1 = {}
                            tmp1["cut_data"] = copy.deepcopy(pvalue_real[head_index:i3])
                            tmp1["cut_time"] = copy.deepcopy(self.t_index[head_index:i3])
                            cut_set.append(tmp1)
                            head_index = i3
                        if i3 == pvalue_real.shape[0]-1:
                            tmp1 = {}
                            tmp1["cut_data"] = copy.deepcopy(pvalue_real[head_index:])
                            tmp1["cut_time"] = copy.deepcopy(self.t_index[head_index:])
                            cut_set.append(tmp1)
                    # 进行glv方程模拟
                    sim_set = []
                    for i4 in range(len(cut_set)):
                        cut_data = copy.deepcopy(cut_set[i4]["cut_data"])
                        cut_time = copy.deepcopy(cut_set[i4]["cut_time"])
                        t_step = cut_time.shape[0]
                        otu_0[0] = copy.deepcopy(cut_data[0])
                        otu_0[1] = np.mean(cut_data)
                        sim_otu = np.zeros(shape=(1,t_step))
                        sim_otu[:,0] = copy.deepcopy(otu_0[1:])

                        for i5 in range(t_step-1):
                            sol = solve_ivp(self.glvModel,[0,cut_time[i5+1]-cut_time[i5]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,cut_time[i5+1]-cut_time[i5]])
                            sim_otu[:,i5+1] = copy.deepcopy(sol.y[:,1][1:])
                            otu_0[:1] = copy.deepcopy(cut_data[i5+1])
                            otu_0[1:] = copy.deepcopy(sol.y[:,1][1:])
                        sim_set.append(sim_otu)
                    
                    finanl_sim = np.zeros(shape=(1,self.t_index.shape[0]))
                    head_index = 0
                    for i6 in range(len(sim_set)):
                        gap = sim_set[i6].shape[1]
                        finanl_sim[:,head_index:head_index+gap] = copy.deepcopy(sim_set[i6])
                        head_index += gap

                    sim_array[epoch,:] = copy.deepcopy(finanl_sim)
            except:
                get_set = False
            if get_set:
                tmp = {}
                tmp["original_real"] = pvalue_real
                tmp["original_sim"] = pvalue_sim
                tmp["sim_datas"] = sim_array
                tmp["Ui"] = Ui_pre
                tmp["Wii"] = Wii_pre
                tmp["Wij"] = Wij_pre
                train_ne_simdata.append(tmp)

        po_distribution = []
        for k in tqdm(range(len(train_po_simdata)),position=0,desc="po_distribution_generating_2",colour='red',ncols=80):
            for i in range(10):
                po_distribution.append(stats.kendalltau(train_po_simdata[k]["original_sim"],train_po_simdata[k]["sim_datas"][i,:])[1])
        po_distribution = -np.log(po_distribution)

        ne_distribution = []
        for k in tqdm(range(len(train_ne_simdata)),position=0,desc="ne_distribution_generating_2",colour='green',ncols=80):
            for i in range(10):
                ne_distribution.append(stats.kendalltau(train_ne_simdata[k]["original_sim"],train_ne_simdata[k]["sim_datas"][i,:])[1])
        ne_distribution = -np.log(ne_distribution)

        mask = np.isinf(po_distribution)
        po_distribution = po_distribution[~mask]
        mask = np.isinf(ne_distribution)
        ne_distribution = ne_distribution[~mask]

        self.po_distribution = po_distribution
        self.ne_distribution = ne_distribution

        dumpf = open(os.path.join(self.result_save_name + '/','po_distribution.txt') ,'wb')         
        pkl.dump(self.po_distribution,dumpf)              
        dumpf.close()

        dumpf = open(os.path.join(self.result_save_name + '/','ne_distribution.txt') ,'wb')         
        pkl.dump(self.ne_distribution,dumpf)              
        dumpf.close()


    def load_Two_Distributions(self,po_position,ne_position):
        self.po_distribution = pkl.load(open(po_position, "rb"))
        self.ne_distribution = pkl.load(open(ne_position, "rb"))
        self.param_po_1,self.param_po_2 = stats.norm.fit(self.po_distribution)
        self.param_ne_1,self.param_ne_2 = stats.norm.fit(self.ne_distribution)

    
    def run_validity_check(self):
        test_simdata = []
        for iq in tqdm(range(self.raw_data.shape[0]),position=0,desc="i",colour='green',ncols=80):
            for jq in tqdm(range(self.raw_data.shape[0]),position=1,desc="j",leave=False,colour='red',ncols=80):
                corr_output = self.model(torch.from_numpy(np.vstack((self.raw_data[jq],self.raw_data[iq]))).float().to("cuda:0"))[1].to("cpu").squeeze().detach().numpy()
                pWii_label = copy.deepcopy(corr_output[1])
                pUi_lable = copy.deepcopy(corr_output[0])
                pWij_label = copy.deepcopy(corr_output[-1])
                pvalue_real = copy.deepcopy(self.raw_data[jq])
                pvalue_sim = copy.deepcopy(self.raw_data[iq])

                sim_array = np.zeros(shape=(10,pvalue_real.shape[0]))
                get_set = True
                try:
                    for epoch in range(10):
                        pUi = pUi_lable+np.random.normal(0,0.01)
                        pWij = copy.deepcopy(pWij_label)
                        #构造相关性矩阵
                        corr_matrix = np.zeros(shape=(2,2))
                        row, col = np.diag_indices_from(corr_matrix)                   
                        corr_matrix[row,col] = np.array([pWii_label+np.random.normal(0,0.01) for i in range(2)])
                        corr_matrix[1,0]=copy.deepcopy(pWij)
                        #构造自增长率
                        growth_rate =  np.array([pUi,pUi])
                        #构造初值
                        otu_0 = np.zeros(shape=2)
                        otu_0[0] = copy.deepcopy(pvalue_real[0])
                        otu_0[1] = np.mean(pvalue_real)

                        # 对输入的拼接数据进行切段
                        head_index = 0
                        cut_set = []
                        for i3 in range(self.t_index.shape[0]):
                            if (self.t_index[i3] < self.t_index[i3-1]) and (i3 != 0):
                                tmp1 = {}
                                tmp1["cut_data"] = copy.deepcopy(pvalue_real[head_index:i3])
                                tmp1["cut_time"] = copy.deepcopy(self.t_index[head_index:i3])
                                cut_set.append(tmp1)
                                head_index = copy.deepcopy(i3)
                            if i3 == pvalue_real.shape[0]-1:
                                tmp1 = {}
                                tmp1["cut_data"] = copy.deepcopy(pvalue_real[head_index:])
                                tmp1["cut_time"] = copy.deepcopy(self.t_index[head_index:])
                                cut_set.append(tmp1)
                        # 进行glv方程模拟
                        sim_set = []
                        for i4 in range(len(cut_set)):
                            cut_data = copy.deepcopy(cut_set[i4]["cut_data"])
                            cut_time = copy.deepcopy(cut_set[i4]["cut_time"])
                            t_step = cut_time.shape[0]
                            otu_0[0] = copy.deepcopy(cut_data[0])
                            otu_0[1] = np.mean(cut_data)
                            sim_otu = np.zeros(shape=(1,t_step))
                            sim_otu[:,0] = copy.deepcopy(otu_0[1:])

                            for i5 in range(t_step-1):
                                sol = solve_ivp(self.glvModel,[0,cut_time[i5+1]-cut_time[i5]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,cut_time[i5+1]-cut_time[i5]])
                                sim_otu[:,i5+1] = copy.deepcopy(sol.y[:,1][1:])
                                otu_0[:1] = copy.deepcopy(cut_data[i5+1])
                                otu_0[1:] = copy.deepcopy(sol.y[:,1][1:])
                            sim_set.append(sim_otu)
                        
                        finanl_sim = np.zeros(shape=(1,self.t_index.shape[0]))
                        head_index = 0
                        for i6 in range(len(sim_set)):
                            gap = sim_set[i6].shape[1]
                            finanl_sim[:,head_index:head_index+gap] = copy.deepcopy(sim_set[i6])
                            head_index += gap

                        sim_array[epoch,:] = copy.deepcopy(finanl_sim)
                except:
                        get_set = False
                if get_set:
                    tmp = {}
                    tmp["original_real"] = pvalue_real
                    tmp["original_sim"] = pvalue_sim
                    tmp["sim_datas"] = sim_array
                    tmp["Ui"] = pUi_lable
                    tmp["Wii"] = pWii_label
                    tmp["Wij"] = pWij_label
                    tmp["iq"] = iq
                    tmp["jq"] = jq
                    test_simdata.append(tmp)
        dumpf = open(os.path.join(self.result_save_name + '/','test_simdata.txt') ,'wb')         
        pkl.dump(test_simdata,dumpf)              
        dumpf.close()

        validity_Wij = np.zeros(shape=(self.raw_data.shape[0],self.raw_data.shape[0]))
        for k in tqdm(range(len(test_simdata)),position=0,colour='green',ncols=80):
            iq = test_simdata[k]["iq"]
            jq = test_simdata[k]["jq"]
            total_logpdf_po = 0
            total_logpdf_ne = 0
            for i in range(10):
                corr = -np.log(stats.kendalltau(tmp[k]["original_sim"],tmp[k]["sim_datas"][i,:])[1])
                total_logpdf_po += stats.norm.logpdf(corr,loc= self.param_po_1,scale=self.param_po_2)
                total_logpdf_ne += stats.norm.logpdf(corr,loc= self.param_ne_1,scale=self.param_ne_2)
            if(total_logpdf_po >= total_logpdf_ne):
                validity_Wij[iq][jq] = test_simdata[k]["Wij"]
        return validity_Wij