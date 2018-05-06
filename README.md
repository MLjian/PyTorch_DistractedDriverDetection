>>>        
# PyTorch_DistractedDriverDetection
## 1 项目概述：<br>
### 1.1 项目来源：<br>
kaggle中的【State Farm Distracted Driver Detection】比赛的实现。https://www.kaggle.com/c/state-farm-distracted-driver-detection<br>
### 1.2 问题概述:<br>
对一张关于驾驶员行为的图片进行分类，共10类：安全驾驶/左手打字/右手打电话/左右打电话/...<br>
### 1.3 数据集下载地址：<br>
https://pan.baidu.com/s/1l2RkXYOAdxuL_jiKWJm1cQ<br>
## 2 程序运行相关：<br>
### 2.1 运行环境：<br>
windows/python3.5/pytorch0.4/visdom<br>
### 2.2 运行前的准备：<br>
s1:下载本repository至本地；<br>
s2:在本目录下，建立文件夹如下,并下载train数据集至data文件夹下：<br>
>>----data<br>
--train<br>
----trained_models<br>
### 2.3运行：<br>
s1.打开visdom后台：*python -m visdom.server*<br>
s2.运行训练文件：*python train.py*<br>
## 3 训练过程可视化：<br>
<div align=center>
<img width="467" height="300" src="https://github.com/MLjian/PyTorch_DistractedDriverDetection/blob/master/training_show/train_loss.png"/>
</div><br>
<div align=center>
<img width="467" height="300" src="https://github.com/MLjian/PyTorch_DistractedDriverDetection/blob/master/training_show/vali_acc.png"/>
</div><br>
  
