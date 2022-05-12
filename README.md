# 一. n皇后问题

## 命令行后加上n即可,默认4皇后

<img src="https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/nqueen.png" alt="nqueen"  />

![nqueen2](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/nqueen2.png)

# 二. Astar最短路径

- ## 单击鼠标左键设置起点和终点

- ## 运行中点击鼠标可暂停

- ## 一次任务结束后按ｒ键可重置

  ![Astar](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/Astar.png)e

# 三.拟合Sin

## 运行效果:

![运行效果](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/%E8%BF%90%E8%A1%8C%E6%95%88%E6%9E%9C.png)

## 训练误差:

![train_loss_ReLU](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/train_loss_ReLU.png)

![train_loss_Sigmoid](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/train_loss_Sigmoid.png)

## 拟合结果:

![result](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/result.png)



## 四.车牌字符识别

### 1.运行

运行python文件或者jupyter notebook,支持tensorboard查看训练过程

```sh
python python CarPlateClassification.py
tensorboard --logdir=logs --port=6006
```

### 2.运行效果:

![运行结果(jupyter)](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/%E8%BF%90%E8%A1%8C%E7%BB%93%E6%9E%9C(jupyter).png)

### 3.数据集

 测试集为作业中20张截图,训练集和验证集是在开源数据集基础上使用换底色制作

[可从此链接下载](https://github.com/Cmoon-cyl/Car-Plate-Character-Dataset.git)

测试集样式:

![测试集(作业的20张图截图)](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/%E6%B5%8B%E8%AF%95%E9%9B%86(%E4%BD%9C%E4%B8%9A%E7%9A%8420%E5%BC%A0%E5%9B%BE%E6%88%AA%E5%9B%BE).png)

训练集样式:

![训练集](https://raw.githubusercontent.com/Cmoon-cyl/Image-Uploader/main/%E8%AE%AD%E7%BB%83%E9%9B%86.png)
