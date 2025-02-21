# GRIDi MLOps Workflow Framework
* 该MLOps框架基于传统机器学习框架sklearn、与深度学习框架PyTorch，集成了包含回归分析、分类算法、聚类算法、LSTM、RNN、Transformer、encoder-decoder架构等主流时序算法能力。
* 框架主要亮点在于整合了从数据处理、数据建模、模型checkpoint、及模型部署应用等模块全流程服务，大幅度减少业务建模的重复性工作，显著提高了建模效率，对于初学者掌握数据建模流程，也是一个非常好的实践模版。

## 1. 数据处理类
### 1.1 algo_box
算法小工具，包含常用的时序统计模块
### 1.2 preprocessing
时序分析数据预处理工作模块
### 1.3 datas
常用时序算法的数据输入模版

## 2. 建模工作流主控模块
### 2.1 mlops
算法核心流程主控模块，集成了包括分类、聚类、决策树模型、深度序列等主流模型训练、保存、部署工作流

## 3 基础算法服务模块
### 3.1 nn
神经网络相关算法，如lstm、rnn等基础服务模块
### 3.2 tasks
包含传统机器学习模型，如kmeans聚类、random forest随机森林、xgboost、lightgbm、catboost树模型等常用算法训练、与测试任务模块

## 4. 分布训练配置模块
### 4.1 baseConfig
基于Ray的模型分布式训练配置