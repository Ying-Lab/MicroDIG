import numpy as np
from scipy.integrate import solve_ivp
from Utils import compute_simulation_parameters_normal,compute_simulation_parameters_uniform
import copy
import random


def Simulation_X(original_data,original_time,original_sim_nums):   # 输入矩阵的行为OTU，列为时间步
    # 深拷贝输入参数
    np_data = copy.deepcopy(original_data)
    t_index = copy.deepcopy(original_time)
    sim_nums = copy.deepcopy(original_sim_nums)
    
    # 给定gLV参数的范围
    Wij_self_min,Wij_self_max,Wij_interact_min,Wij_interact_max,Ui_min,Ui_max = compute_simulation_parameters_uniform(np_data,t_index)

    # 构造参数矩阵
    real_otu_nums = np_data.shape[0]
    def rand():
        return np.random.uniform(Wij_interact_min, Wij_interact_max) if np.random.randint(0, 2) else -np.random.uniform(Wij_interact_min, Wij_interact_max)
    
    parameters = []
    for _ in range(sim_nums//6):
        parameter = {}
        parameter['Wij'] = rand()
        parameter['Ui'] = np.abs(np.random.uniform(Ui_loc, Ui_scale))
        parameter['index'] = np.random.randint(real_otu_nums)
        parameters.append(parameter)
    for i1 in range(sim_nums//6):
        if np.random.randint(0, 2):
            for _ in range(5):
                parameter = {}
                parameter['Wij'] = rand()
                parameter['Ui'] = copy.deepcopy(parameters[i1]['Ui'])
                parameter['index'] = copy.deepcopy(parameters[i1]['index'])
                parameters.append(parameter)
        else:
            for _ in range(5):
                parameter = {}
                parameter['Wij'] = copy.deepcopy(parameters[i1]['Wij'])
                parameter['Ui'] = np.abs(np.random.uniform(Ui_loc, Ui_scale))
                parameter['index'] = copy.deepcopy(parameters[i1]['index'])
                parameters.append(parameter)
    random.shuffle(parameters)

    corr_matrix = np.zeros(shape=(real_otu_nums+sim_nums,real_otu_nums+sim_nums)) 
    row, col = np.diag_indices_from(corr_matrix)                   
    corr_matrix[row,col] = np.array([-np.abs(np.random.uniform(Wij_self_min, Wij_self_max)) for i in range(real_otu_nums+sim_nums)])

    otu_0_index = np.zeros(shape=real_otu_nums+sim_nums)   
    otu_0_index[:real_otu_nums] = -1 # -1代表无配对关系，是real的部分

    growth_rate =  np.zeros(shape=real_otu_nums+sim_nums)
    growth_rate[:real_otu_nums] = np.array([np.abs(np.random.uniform(Ui_loc, Ui_scale)) for i in range(real_otu_nums)])

    for i2 in range(real_otu_nums,real_otu_nums+sim_nums):
        parameter = copy.deepcopy(parameters[i2-real_otu_nums])
        corr_matrix[i2,parameter['index']] = copy.deepcopy(parameter['Wij'])
        growth_rate[i2] = copy.deepcopy(parameter['Ui'])
        otu_0_index[i2] = copy.deepcopy(parameter['index'])  # 值代表和其配对的real otu

    # 对输入的拼接数据进行切段
    head_index = 0
    cut_set = []
    for i3 in range(t_index.shape[0]):
        if (t_index[i3] < t_index[i3-1]) and (i3 != 0):
            tmp1 = {}
            tmp1["cut_data"] = copy.deepcopy(np_data[:,head_index:i3])
            tmp1["cut_time"] = copy.deepcopy(t_index[head_index:i3])
            cut_set.append(tmp1)
            head_index = copy.deepcopy(i3)
        if i3 == np_data.shape[1]-1:
            tmp1 = {}
            tmp1["cut_data"] = copy.deepcopy(np_data[:,head_index:])
            tmp1["cut_time"] = copy.deepcopy(t_index[head_index:])
            cut_set.append(tmp1)

    # 分段模拟数据
    def glvModel(t,x0,growth_rate,corr_matrix):
        dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
        return dx
    sim_set = []
    for i4 in range(len(cut_set)):
        cut_data = copy.deepcopy(cut_set[i4]["cut_data"])
        cut_time = copy.deepcopy(cut_set[i4]["cut_time"])
        t_step = cut_time.shape[0]
        otu_0 = np.zeros(shape=real_otu_nums+sim_nums)
        otu_0[:real_otu_nums] = copy.deepcopy(cut_data[:,0])
        for i8 in range(real_otu_nums,real_otu_nums+sim_nums):
            otu_0[i8] = np.mean(cut_data[otu_0_index[i8].astype(int)])


        sim_otu = np.zeros(shape=(sim_nums,t_step))
        sim_otu[:,0] = copy.deepcopy(otu_0[real_otu_nums:])

        for i5 in range(t_step-1):
            sol = solve_ivp(glvModel,[0,cut_time[i5+1]-cut_time[i5]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,cut_time[i5+1]-cut_time[i5]])
            sim_otu[:,i5+1] = copy.deepcopy(sol.y[:,1][real_otu_nums:])
            otu_0[:real_otu_nums] = copy.deepcopy(cut_data[:,i5+1])
            otu_0[real_otu_nums:] = copy.deepcopy(sol.y[:,1][real_otu_nums:])
        sim_set.append(sim_otu)
    
    finanl_sim = np.zeros(shape=(sim_nums,t_index.shape[0]))
    head_index = 0
    for i6 in range(len(sim_set)):
        gap = sim_set[i6].shape[1]
        finanl_sim[:,head_index:head_index+gap] = copy.deepcopy(sim_set[i6])
        head_index += gap
    
    # 整合输出
    pair_set = []
    for i7 in range(sim_nums):
        temp = {}
        temp['real']=copy.deepcopy(np_data[parameters[i7]['index'],:].squeeze())
        no_real_index = list(range(real_otu_nums))
        no_real_index.remove(parameters[i7]['index'])
        temp['no_real']=copy.deepcopy(np_data[[np.random.choice(no_real_index)],:].squeeze())
        temp['sim']=copy.deepcopy(finanl_sim[i7,:])
        temp['growth_rate'] = copy.deepcopy(growth_rate[real_otu_nums+i7])
        temp['Wii']=copy.deepcopy(corr_matrix[i7+real_otu_nums,i7+real_otu_nums])
        temp['Wij']=copy.deepcopy(corr_matrix[i7+real_otu_nums,parameters[i7]['index']])
        pair_set.append(temp)
    return pair_set



#-------------------------------------------------------------------------------------------------------------------
Time_Series_Max_Judgment = 0.11
Max_Interact = 6
Min_Interact = -6
Sim_Otu_Number = 6000

def Simulation_OTU_iter_uniform_Final(np_data,sim_nums,t_index):
    def Simulation_OTU_iter_uniform_Robust(real_data,np_data,sim_nums,t_index,pair_set):
        def Simulation_OTU_iter_uniform(real_data,np_data,sim_nums,t_index,pair_set):   # 输入矩阵的行为OTU，列为时间步
            otu_nums = np_data.shape[0]
            t_step = np_data.shape[1]
            
            #估计近似参数
            Wij_self_min,Wij_self_max,Wij_interact_min,Wij_interact_max,Ui_min,Ui_max = compute_simulation_parameters_uniform(real_data,t_index)
            if Wij_interact_max>Max_Interact:
                Wij_interact_max=Max_Interact
            if Wij_interact_min<Min_Interact:
                Wij_interact_min=Min_Interact

            # 构造模拟的相关性矩阵 （结构为（real + sim）*（real + sim））
            corr_matrix = np.zeros(shape=(otu_nums+sim_nums,otu_nums+sim_nums)) 
            # self interactions
            row, col = np.diag_indices_from(corr_matrix)                   
            corr_matrix[row,col] = np.array([-np.abs(np.random.uniform(Wij_self_min, Wij_self_max)) for i in range(otu_nums+sim_nums)])
            # other interaction
            def rand():
                return np.random.uniform(Wij_interact_min, Wij_interact_max) if np.random.randint(0, 2) else -np.random.uniform(Wij_interact_min, Wij_interact_max)
            index_list = np.random.randint(otu_nums,size=sim_nums)
            for i in range(otu_nums,otu_nums+sim_nums):
                corr_matrix[i,index_list[i-otu_nums]] = rand()
                
            # 构造初始值(结构为（real + sim）)
            otu_0 = np.zeros(shape=otu_nums+sim_nums)   
            otu_0[:otu_nums] = np_data[:,0]
            otu_0[otu_nums:] = np.mean(np_data[index_list,:],axis=1)

            # 构造自增长率
            growth_rate =  np.array([np.abs(np.random.uniform(Ui_min, Ui_max)) for i in range(otu_nums+sim_nums)])
            for i in range(otu_nums,otu_nums+sim_nums):
                if (abs(np_data[index_list[i-otu_nums],:]).max())<Time_Series_Max_Judgment:
                    growth_rate[i]=0.01 

            # 进行glv方程模拟
            def glvModel(t,x0,growth_rate,corr_matrix):
                dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
                return dx
            sim_otu = np.zeros(shape=(sim_nums,t_step))
            sim_otu[:,0] = otu_0[otu_nums:]
            for i in range(t_step-1):
                if(t_index[i+1]<t_index[i]):
                    sim_otu[:,i+1] = sim_otu[:,0]
                    otu_0[:otu_nums] = np_data[:,i+1]
                    otu_0[otu_nums:] = sim_otu[:,i+1]
                else:
                    sol = solve_ivp(glvModel,[0,t_index[i+1]-t_index[i]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,t_index[i+1]-t_index[i]])
                    sim_otu[:,i+1] = sol.y[:,1][otu_nums:]
                    otu_0[:otu_nums] = np_data[:,i+1]
                    otu_0[otu_nums:] = sim_otu[:,i+1]

            # 整合输出
            for i in range(sim_nums):
                temp = {}
                temp['real']=np_data[[index_list[i]],:].squeeze()
                temp['no_real']=np_data[[np.random.choice(index_list[index_list!=index_list[i]])],:].squeeze()
                temp['sim']=sim_otu[i,:]
                temp['growth_rate'] = growth_rate[otu_nums+i]
                temp['Wii']=corr_matrix[i+otu_nums,i+otu_nums]
                temp['Wij']=corr_matrix[i+otu_nums,index_list[i]]
                pair_set.append(temp)
            return sim_otu
        
        get_set = False
        while not get_set:
            try:
                sim_otu = Simulation_OTU_iter_uniform(real_data,np_data,sim_nums,t_index,pair_set)
                get_set = True
            except:
                pass
        return sim_otu
    pair_set = []
    sim_otu = Simulation_OTU_iter_uniform_Robust(np_data,np_data,100,t_index,pair_set)
    sim_otu = Simulation_OTU_iter_uniform_Robust(np_data,sim_otu,sim_nums-100,t_index,pair_set)
    return pair_set


def Simulation_OTU_uniform_Robust(np_data,sim_nums,t_index):
    def Simulation_OTU_uniform(np_data,sim_nums,t_index):   # 输入矩阵的行为OTU，列为时间步
        otu_nums = np_data.shape[0]
        t_step = np_data.shape[1]
        
        #估计近似参数
        Wij_self_min,Wij_self_max,Wij_interact_min,Wij_interact_max,Ui_min,Ui_max = compute_simulation_parameters_uniform(np_data,t_index)
        if Wij_interact_max>Max_Interact:
            Wij_interact_max=Max_Interact
        if Wij_interact_min<Min_Interact:
            Wij_interact_min=Min_Interact

        # 构造模拟的相关性矩阵 （结构为（real + sim）*（real + sim））
        corr_matrix = np.zeros(shape=(otu_nums+sim_nums,otu_nums+sim_nums)) 
        # self interactions
        row, col = np.diag_indices_from(corr_matrix)                   
        corr_matrix[row,col] = np.array([-np.abs(np.random.uniform(Wij_self_min, Wij_self_max)) for i in range(otu_nums+sim_nums)])
        # other interaction
        def rand():
            return np.random.uniform(Wij_interact_min, Wij_interact_max) if np.random.randint(0, 2) else -np.random.uniform(Wij_interact_min, Wij_interact_max)
        index_list = np.random.randint(otu_nums,size=sim_nums)
        for i in range(otu_nums,otu_nums+sim_nums):
            corr_matrix[i,index_list[i-otu_nums]] = rand()
            
        # 构造初始值(结构为（real + sim）)
        otu_0 = np.zeros(shape=otu_nums+sim_nums)   
        otu_0[:otu_nums] = np_data[:,0]
        otu_0[otu_nums:] = np.mean(np_data[index_list,:],axis=1)

        # 构造自增长率
        growth_rate =  np.array([np.abs(np.random.uniform(Ui_min, Ui_max)) for i in range(otu_nums+sim_nums)])
        for i in range(otu_nums,otu_nums+sim_nums):
            if (abs(np_data[index_list[i-otu_nums],:]).max())<Time_Series_Max_Judgment:
                growth_rate[i]=0.01 

        # 进行glv方程模拟
        def glvModel(t,x0,growth_rate,corr_matrix):
            dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
            return dx
        sim_otu = np.zeros(shape=(sim_nums,t_step))
        sim_otu[:,0] = otu_0[otu_nums:]
        for i in range(t_step-1):
            if(t_index[i+1]<t_index[i]):
                sim_otu[:,i+1] = sim_otu[:,0]
                otu_0[:otu_nums] = np_data[:,i+1]
                otu_0[otu_nums:] = sim_otu[:,i+1]
            else:
                sol = solve_ivp(glvModel,[0,t_index[i+1]-t_index[i]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,t_index[i+1]-t_index[i]])
                sim_otu[:,i+1] = sol.y[:,1][otu_nums:]
                otu_0[:otu_nums] = np_data[:,i+1]
                otu_0[otu_nums:] = sim_otu[:,i+1]

        # 整合输出
        pair_set = []
        for i in range(sim_nums):
            temp = {}
            temp['real']=np_data[[index_list[i]],:].squeeze()
            temp['no_real']=np_data[[np.random.choice(index_list[index_list!=index_list[i]])],:].squeeze()
            temp['sim']=sim_otu[i,:]
            temp['growth_rate'] = growth_rate[otu_nums+i]
            temp['Wii']=corr_matrix[i+otu_nums,i+otu_nums]
            temp['Wij']=corr_matrix[i+otu_nums,index_list[i]]
            pair_set.append(temp)
        return pair_set
    get_set = False
    while not get_set:
        try:
            pair_set = Simulation_OTU_uniform(np_data,sim_nums,t_index)
            get_set = True
        except:
            pass
    return pair_set


def Simulation_OTU(np_data,sim_nums,t_index):   # 输入矩阵的行为OTU，列为时间步
    otu_nums = np_data.shape[0]
    t_step = np_data.shape[1]
    
    #估计近似参数
    Wij_self_mean,Wij_self_var,Wij_interact_var,Ui_mean,Ui_var,initial_mean,initial_var = compute_simulation_parameters_normal(np_data,t_index)

    # 构造模拟的相关性矩阵 （结构为（real + sim）*（real + sim））
    corr_matrix = np.zeros(shape=(otu_nums+sim_nums,otu_nums+sim_nums)) 
    # self interactions
    row, col = np.diag_indices_from(corr_matrix)                   
    corr_matrix[row,col] = np.array([-np.abs(np.random.normal(Wij_self_mean, np.sqrt(Wij_self_var))) for i in range(otu_nums+sim_nums)])
    # other interaction
    def rand():
        return np.random.normal(0, np.sqrt(Wij_interact_var)) if np.random.randint(0, 2) else -np.random.normal(0, np.sqrt(Wij_interact_var))
    index_list = np.random.randint(otu_nums,size=sim_nums)
    for i in range(otu_nums,otu_nums+sim_nums):
        corr_matrix[i,index_list[i-otu_nums]] = rand()
        
    # 构造初始值(结构为（real + sim）)
    otu_0 = np.zeros(shape=otu_nums+sim_nums)   
    otu_0[:otu_nums] = np_data[:,0]
    otu_0[otu_nums:] = np.mean(np_data[index_list,:],axis=1)
    # initial_list = np_data[:,0]
    # initial_list = initial_list[initial_list != 0]
    # otu_0[otu_nums:] = np.random.choice(a=initial_list,size=sim_nums,replace=True)

    # 构造自增长率
    growth_rate =  np.array([np.abs(np.random.normal(Ui_mean, np.sqrt(Ui_var))) for i in range(otu_nums+sim_nums)]) 
    for i in range(otu_nums,otu_nums+sim_nums):
        if (abs(np_data[index_list[i-otu_nums],:]).max())<Time_Series_Max_Judgment:
            growth_rate[i]=0.01

    # 进行glv方程模拟
    def glvModel(t,x0,growth_rate,corr_matrix):
        dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
        return dx
    sim_otu = np.zeros(shape=(sim_nums,t_step))
    sim_otu[:,0] = otu_0[otu_nums:]
    for i in range(t_step-1):
        if(t_index[i+1]<t_index[i]):
            sim_otu[:,i+1] = sim_otu[:,0]
            otu_0[:otu_nums] = np_data[:,i+1]
            otu_0[otu_nums:] = sim_otu[:,i+1]
        else:
            sol = solve_ivp(glvModel,[0,t_index[i+1]-t_index[i]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,t_index[i+1]-t_index[i]])
            sim_otu[:,i+1] = sol.y[:,1][otu_nums:]
            otu_0[:otu_nums] = np_data[:,i+1]
            otu_0[otu_nums:] = sim_otu[:,i+1]

    # 整合输出
    pair_set = []
    for i in range(sim_nums):
        temp = {}
        temp['real']=np_data[[index_list[i]],:].squeeze()
        temp['no_real']=np_data[[np.random.choice(index_list[index_list!=index_list[i]])],:].squeeze()
        temp['sim']=sim_otu[i,:]
        temp['growth_rate'] = growth_rate[otu_nums+i]
        temp['Wii']=corr_matrix[i+otu_nums,i+otu_nums]
        temp['Wij']=corr_matrix[i+otu_nums,index_list[i]]
        pair_set.append(temp)
    return pair_set

def Simulation_OTU_Robust(np_data,sim_nums,t_index):
    get_set = False
    while not get_set:
        try:
            pair_set = Simulation_OTU(np_data,sim_nums,t_index)
            get_set = True
        except:
            pass
    return pair_set

def Simulation_OTU_normal_Robust(np_data,sim_nums,t_index):
    def Simulation_OTU_normal(np_data,sim_nums,t_index):   # 输入矩阵的行为OTU，列为时间步
        otu_nums = np_data.shape[0]
        t_step = np_data.shape[1]
        
        #估计近似参数
        Wij_self_mean,Wij_self_var,Wij_interact_var,Ui_mean,Ui_var,initial_mean,initial_var = compute_simulation_parameters_normal(np_data,t_index)

        # 构造模拟的相关性矩阵 （结构为（real + sim）*（real + sim））
        corr_matrix = np.zeros(shape=(otu_nums+sim_nums,otu_nums+sim_nums)) 
        # self interactions
        row, col = np.diag_indices_from(corr_matrix)                   
        corr_matrix[row,col] = np.array([-np.abs(np.random.normal(Wij_self_mean, np.sqrt(Wij_self_var))) for i in range(otu_nums+sim_nums)])
        # other interaction
        def rand():
            return np.random.normal(0, np.sqrt(Wij_interact_var)) if np.random.randint(0, 2) else -np.random.normal(0, np.sqrt(Wij_interact_var))
        index_list = np.random.randint(otu_nums,size=sim_nums)
        for i in range(otu_nums,otu_nums+sim_nums):
            corr_matrix[i,index_list[i-otu_nums]] = rand()
            
        # 构造初始值(结构为（real + sim）)
        otu_0 = np.zeros(shape=otu_nums+sim_nums)   
        otu_0[:otu_nums] = np_data[:,0]
        otu_0[otu_nums:] = np.mean(np_data[index_list,:],axis=1)

        # 构造自增长率
        growth_rate =  np.array([np.abs(np.random.normal(Ui_mean, np.sqrt(Ui_var))) for i in range(otu_nums+sim_nums)]) 
        for i in range(otu_nums,otu_nums+sim_nums):
            if (abs(np_data[index_list[i-otu_nums],:]).max())<Time_Series_Max_Judgment:
                growth_rate[i]=0.01

        # 进行glv方程模拟
        def glvModel(t,x0,growth_rate,corr_matrix):
            dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
            return dx
        sim_otu = np.zeros(shape=(sim_nums,t_step))
        sim_otu[:,0] = otu_0[otu_nums:]
        for i in range(t_step-1):
            if(t_index[i+1]<t_index[i]):
                sim_otu[:,i+1] = sim_otu[:,0]
                otu_0[:otu_nums] = np_data[:,i+1]
                otu_0[otu_nums:] = sim_otu[:,i+1]
            else:
                sol = solve_ivp(glvModel,[0,t_index[i+1]-t_index[i]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,t_index[i+1]-t_index[i]])
                sim_otu[:,i+1] = sol.y[:,1][otu_nums:]
                otu_0[:otu_nums] = np_data[:,i+1]
                otu_0[otu_nums:] = sim_otu[:,i+1]

        # 整合输出
        pair_set = []
        for i in range(sim_nums):
            temp = {}
            temp['real']=np_data[[index_list[i]],:].squeeze()
            temp['no_real']=np_data[[np.random.choice(index_list[index_list!=index_list[i]])],:].squeeze()
            temp['sim']=sim_otu[i,:]
            temp['growth_rate'] = growth_rate[otu_nums+i]
            temp['Wii']=corr_matrix[i+otu_nums,i+otu_nums]
            temp['Wij']=corr_matrix[i+otu_nums,index_list[i]]
            pair_set.append(temp)
        return pair_set
    get_set = False
    while not get_set:
        try:
            pair_set = Simulation_OTU_normal(np_data,sim_nums,t_index)
            get_set = True
        except:
            pass
    return pair_set



def Simulation_multi_uniform_Robust(np_data,sim_nums,t_index,asso_nums):
    def Simulation_multi_uniform(np_data,sim_nums,t_index):   # 输入矩阵的行为OTU，列为时间步
        otu_nums = np_data.shape[0]
        t_step = np_data.shape[1]
        
        #估计近似参数
        Wij_self_min,Wij_self_max,Wij_interact_min,Wij_interact_max,Ui_min,Ui_max = compute_simulation_parameters_uniform(np_data,t_index)
        if Wij_interact_max>Max_Interact:
            Wij_interact_max=Max_Interact
        if Wij_interact_min<Min_Interact:
            Wij_interact_min=Min_Interact

        # 构造模拟的相关性矩阵 （结构为（real + sim）*（real + sim））
        corr_matrix = np.zeros(shape=(otu_nums+sim_nums,otu_nums+sim_nums)) 
        # self interactions
        row, col = np.diag_indices_from(corr_matrix)                   
        corr_matrix[row,col] = np.array([-np.abs(np.random.uniform(Wij_self_min, Wij_self_max)) for i in range(otu_nums+sim_nums)])
        # other interaction
        def rand():
            return np.random.uniform(Wij_interact_min, Wij_interact_max) if np.random.randint(0, 2) else -np.random.uniform(Wij_interact_min, Wij_interact_max)
        index_list = []
        for i in range(otu_nums,otu_nums+sim_nums):
            corr_list = np.random.randint(otu_nums,size=asso_nums)
            index_list.append(corr_list)
            for j in range(asso_nums):
                corr_matrix[i,corr_list[j]] = rand()
            
        # 构造初始值(结构为（real + sim）)
        otu_0 = np.zeros(shape=otu_nums+sim_nums)   
        otu_0[:otu_nums] = np_data[:,0]
        otu_0[otu_nums:] = np.mean(np.mean(np_data[index_list,:],axis=2),axis=1)

        # 构造自增长率
        growth_rate =  np.array([np.abs(np.random.uniform(Ui_min, Ui_max)) for i in range(otu_nums+sim_nums)])
        for i in range(otu_nums,otu_nums+sim_nums):
            if (abs(np_data[index_list[0],:]).max(axis=1).min())<Time_Series_Max_Judgment:
                growth_rate[i]=0.01 

        # 进行glv方程模拟
        def glvModel(t,x0,growth_rate,corr_matrix):
            dx = x0 * (growth_rate + np.dot(corr_matrix,x0))
            return dx
        sim_otu = np.zeros(shape=(sim_nums,t_step))
        sim_otu[:,0] = otu_0[otu_nums:]
        for i in range(t_step-1):
            if(t_index[i+1]<t_index[i]):
                sim_otu[:,i+1] = sim_otu[:,0]
                otu_0[:otu_nums] = np_data[:,i+1]
                otu_0[otu_nums:] = sim_otu[:,i+1]
            else:
                sol = solve_ivp(glvModel,[0,t_index[i+1]-t_index[i]],otu_0,args=(growth_rate,corr_matrix),t_eval=[0,t_index[i+1]-t_index[i]])
                sim_otu[:,i+1] = sol.y[:,1][otu_nums:]
                otu_0[:otu_nums] = np_data[:,i+1]
                otu_0[otu_nums:] = sim_otu[:,i+1]

        # 整合输出
        pair_set = []
        for i in range(sim_nums):
            for j in range(asso_nums):
                temp = {}
                temp['real']=np_data[index_list[i][j],:]
                temp['no_real']=np_data[np.random.choice(np.delete(np.arange(otu_nums),index_list[i])),:]
                temp['sim']=sim_otu[i,:]
                temp['growth_rate'] = growth_rate[otu_nums+i]
                temp['Wii']=corr_matrix[i+otu_nums,i+otu_nums]
                temp['Wij']=corr_matrix[i+otu_nums,index_list[i][j]]
                pair_set.append(temp)
        return pair_set
    get_set = False
    while not get_set:
        try:
            pair_set = Simulation_multi_uniform(np_data,sim_nums,t_index)
            get_set = True
        except:
            pass
    return pair_set