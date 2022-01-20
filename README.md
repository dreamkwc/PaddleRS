# A飞桨常规赛：dreamkwc的团队 - 12月第2名方案

## 项目描述
> 基于PaddleSeg构建的，使用DeepLabV3P模型，backbone替换为了MobileNetv2.

## 项目结构
```
-|data
-|PaddleSeg
-|work
-README.MD
-train.ipynb
-predict.ipynb
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)  
B：将比赛数据集压缩文件放在data文件夹中。  
C: 修改train.ipynb中的压缩文件名，依次运行每个cell即可训练。  
D: 运行predict.ipynb文件中的每个cell即可完成预测推理，结果保存在Output/result/result文件中。  
E: 配置文件为PaddleSeg/configs/deeplabv3p/deeplabv3p_mobilenetv2_g.yml  

## 说明
没有保留提交时的checkpoint文件。
