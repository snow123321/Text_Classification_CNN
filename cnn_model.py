# -*- coding: utf-8 -*-

'''-------------------------CNN配置参数及模型-----------------------------'''

import tensorflow as tf

class TextCNNConfig(object):
    embedding_dim = 64   #词向量维度
    seq_length = 600   #序列长度
    num_classes = 10   #类别数
    
    #第一个卷积层参数
    conv1_num_filters = 128   #卷积核数目
    conv1_kernel_size = 2   #卷积核尺寸，可以相当于是2-gram
    
    #第二个卷积层参数
    conv2_num_filters = 64
    conv2_kernel_size = 3   #卷积核尺寸，可以相当于是3-gram
    
    vocab_size = 5000   #词汇表大小
    
    hidden_dim = 128   #全连接层神经元
    
    dropout_keep_prob = 0.5   #drop保留比例
    learning_rate = 1e-3   #学习率
    
    batch_size = 64   #每批训练大小
    n_epochs = 5   #总迭代轮次
    
    print_per_batch = 100   #每多少轮输出一次结果
    save_per_batch = 1000   #每多少轮存入tensorboard
    
    
    
'''文本分类，CNN模型'''
class TextCNN(object):
    def __init__(self,config):
        self.config = config
        
        self.input_x = tf.placeholder(tf.int32,[None,self.config.seq_length],name = "input_x")
        self.input_y = tf.placeholder(tf.int32,[None,self.config.num_classes],name = "input_y")
        self.dropout_prob = tf.placeholder(tf.float32,name = "dropout_prob")
        
        self.cnn()
        
    def cnn(self):
        #词向量映射
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",[self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding,self.input_x)
            
        #cnn模型
        with tf.name_scope("cnn"):
            #第一层卷积  conv1
            conv1 = tf.layers.conv1d(inputs=embedding_inputs,
                                     filters=self.config.conv1_num_filters,
                                     kernel_size=self.config.conv1_kernel_size,
                                     padding = "SAME",
                                     name = "conv1")
            
            #conv1形状[batch_size,seq_length,conv1_num_filters]
            #第二层卷积  conv2
            conv2 = tf.layers.conv1d(inputs=conv1,
                                     filters=self.config.conv2_num_filters,
                                     kernel_size=self.config.conv2_kernel_size,
                                     padding = "SAME",
                                     name = "conv2")

            #池化层  最大池化  max_pooling layer
            #输入conv2是3维张量，大小[batch_size,seq_length,conv2_num_filters];
            #reduction_indices=[1]表示按第1维取最大值，reduction_indices=[0]表示按第0维取最大值
            #最后的pool形状[batch_size,conv2_num_filters]
            pool = tf.reduce_max(conv2,reduction_indices=[1],name = "pooling")   
            print("pool.shape：",pool.shape)
            
            #全连接层，后接dropout及relu激活
            full_c = tf.layers.dense(pool,units=self.config.hidden_dim,name = "full_c")
            full_c = tf.contrib.layers.dropout(full_c,self.dropout_prob)
            full_c = tf.nn.relu(full_c,name = "relu")
            
            #输出层
            self.logits = tf.layers.dense(full_c,units=self.config.num_classes,name = "logits")
            self.y_pred = tf.argmax(self.logits,axis=1)   #预测类别
            
        #训练优化    
        with tf.name_scope("training_op"):
            #交叉熵和损失函数
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss = tf.reduce_mean(xentropy)
            
            #优化
            optimiaer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate)
            self.optim = optimiaer.minimize(self.loss)
            
        #计算准确率
        with tf.name_scope("accuracy"):
            correct = tf.equal(tf.argmax(self.input_y,axis=1),self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct,tf.float32))
            







