# MicroDIG
## 概述
MicroDIG是一款基于微生物群落时序丰度数据训练估计群落间作用关系的作用关系网络构建工具。MicroDIG利用GLV(Generalized Lotka-Volterra)方程进行数据模拟与增强，通过神经网络自编码器学习物种间的有向作用强度，并设计有效性检验策略对有向作用强度进行统计校验，以获得具有高置信度的物种间有向作用关系。
## 安装
MicroDIG通过Python语言实现,使用MicroDIG之前需要安装对应的Python环境：
```
git clone https://github.com/Ying-Lab/MicroDIG.git
pip install -r requirements.txt
``` 

## 运行MicroDIG框架
``` 
microDIG = MicroDIG(input_data,input_t_index,isNormal,result_save_name)
microDIG.Generate_Enhanced_data()
microDIG.Get_trainAndval_set()
microDIG.train_model()
microDIG.get_Wij_without_check()


--input_data        时序数据输入
--input_t_index     时间点信息输入
--isNormal          是否归一化
--result_save_name  结果保存路径
```
## 使用示例
```
data = pd.read_csv("test_Data/R_sim_data.csv",index_col=0)
t_index = np.array(data.iloc[-1,:])
data = np.array(data.iloc[:-1,:])
microDIG = MicroDIG(data,t_index,False,"test_result_save_dir")
microDIG.Generate_Enhanced_data()
microDIG.Get_trainAndval_set()
microDIG.train_model()
microDIG.get_Wij_without_check()
```
