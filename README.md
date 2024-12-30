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
python ViTax.py [--contigs INPUT_FA] [--out OUTPUT_TXT] 
--contigs INPUT_FA   input fasta file
--out OUTPUT_TXT     The output csv file (prediction_output.txt default) 
--confidence         The confidence threshold of the prediction (0.6 default)  
--rc                 include reverse complement prediction (default True)
--window_size        The sliding window size (default 400)
```
## 使用示例
```
python ViTax.py --contigs test.fa --out prediction_output.txt --confidence 0.6 --rc False --window_size 400
```
