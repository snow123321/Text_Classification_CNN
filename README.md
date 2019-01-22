# 学习笔记：用CNN进行中文文本分类
本文是基于TensorFlow，使用CNN进行的中文文本分类。<br>
## 环境
TensorFlow<br>
Python3<br>
## 数据集
使用THUCNews的一个子集进行训练与测试（由于数据集太大，无法上传到Github，可自行下载），数据集划分如下：<br>
训练集cnews.train.txt   50000条<br>
验证集cnews.val.txt   5000条<br>
测试集cnews.test.txt   10000条<br>
共分为10个类别："体育","财经","房产","家居","教育","科技","时尚","时政","游戏","娱乐"。<br>
cnews.vocab.txt为词汇表，字符级，大小为5000，根据频次选择的。<br>
## 文件说明
data_loader.py：数据预处理文件<br>
cnn_model.py：基本配置参数及CNN模型文件<br>
run_CNN.py：主函数文件<br>

## CNN模型
卷积层有两层，分别对应两种卷积核[2,3]。第一层kernel_size=2，卷积核数量num_filters=128；第二层kernel_size=3，卷积核数量num_filters=64。输入embedding_dim=64，字符级词向量（char embedding），调用tensorflow的API实现，一开始随机初始化，在训练过程中一起训练。处理的是文本，因此用的是conv1d卷积。<br>
这里说一下每一层输出张量大小（忽略batch_size这一维）<br>
* 输入层是词向量，形状为[seq_length, embedding_dim]<br>
* conv1：[seq_length, conv1_filters]，也就是[600,128] （padding=”SAME”，即在卷积时用0进行padding，所以卷积后的句子长度和输入一样）<br>
* conv2：[seq_length, conv2_filters]，也就是[600,64]<br>
* max_pooling：是在每一个feature_map维度上取最大值，最后是大小为64的一维张量<br>
## 结果
<img src="https://github.com/snow123321/Text_Classification_CNN/blob/master/images/picture_1.png" width="400" height="350"><br>

## 学习过程中遇到的问题
### 1、卷积层的维度
* 一个卷积核对应一个feature_map，一个大小为3的卷积核，若filters=100，相当于是100个不同的卷积核，对应100个feature_map。若有大小分别为[2,3,4]三种卷积核，则最后会得到300个feature_map。<br>
* tf.layers.conv1d()这个函数是在一维上取卷积，若kernel_size=3，embedding_dim=64，可以理解为是一个大小为[3,64]的卷积核。<br>
* 假设句子长度为20，卷积核数量（filters）为50，输入20\*64经过一个3\*64的卷积核卷积后得到一个20\*1的向量(padding="SAME")，因为有50个卷积核，所以最终得到50个20\*1的向量，即最后是20\*50。<br>
### 2、池化层
池化层是在feature_map维度上进行池化，所以经过池化层后，一个长度为20的句子最后只有一个值，假设有100个feature_map，最后就是一个100维的向量。<br>



