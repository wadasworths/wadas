## wadas Tech Wiki

## Faster Rcnn tf

项目代码：https://github.com/endernewton/tf-faster-rcnn 

搭建的环境是Ubuntu：16.04 Python3.6 Tensorflow 1.9 GPU模式。在VGG16数据集上训练。训练结果打包成pb文件，采用
tensorflow/serving 和 flask Restful提供服务。

对于训练好的tensorflow模型，有三种常用的打包方式，第一种，打包成ckpt；第二种打包成fronze_model；第三种打包成saved_model.


### Faster Rcnn学习

一文读懂Faster Rcnn：https://zhuanlan.zhihu.com/p/31426458

#### CNN学习 

https://www.jianshu.com/p/da0c4cc76a06

CNN卷积神经网络，CNN一共有卷积层（CONV）、ReLU层（ReLU）、池化层（Pooling）、全连接层（FC（Full Connection））

CONV层：
ReLU层：激活函数，对于每一个维度经过ReLU函数输出即可。不改变数据的空间尺度。
