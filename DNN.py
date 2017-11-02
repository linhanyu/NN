import tensorflow as tf
import numpy as np
import pandas as pd

class dnn_classifier:
    def __init__(self):
        self.layer = []
        self.sess = None
        self.xs = None
    def add_layer(self,inputs,in_size,out_size,activation_function=None):
        # 添加层
        weights = tf.Variable(tf.random_normal([in_size,out_size]))
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)

        propogate = tf.matmul(inputs,weights) + biases

        if activation_function is None:
            outputs = propogate
        else:
            outputs = activation_function(propogate)

        self.layer.append(outputs)
        return outputs

    def fit(self,X,y):
        xs = tf.placeholder(tf.float32,[None,4])
        ys = tf.placeholder(tf.float32,[None,3])
        # 1个隐含层
        self.add_layer(xs,4,5,activation_function=tf.nn.relu)
        self.add_layer(self.layer[0],5,3,activation_function=tf.nn.softmax)
        # self.add_layer(xs,4,3,activation_function=tf.nn.softmax)


        # 交叉熵损失
        # loss =
        loss = -tf.reduce_sum(ys*tf.log(self.layer[-1] + 1e-10))

        # loss = ys*tf.log(self.layer[-1] + 1e-10)

        # 根据交叉熵梯度进行梯度下降
        train_step = tf.train.GradientDescentOptimizer(4*1e-4).minimize(loss)

        #训练
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        # batch_gradient_decent
        for i in range(10000):
            # print(sess.run(loss,feed_dict={xs:X, ys:y}))
            # print(sess.run(train_step,feed_dict={ys:y,xs:X}))
            sess.run(train_step,feed_dict={xs:X,ys:y})
            if i % 1000:
                print(sess.run(loss,feed_dict={ys:y,xs:X}))
        self.sess = sess
        self.xs = xs
    def predict(self,X):
        return np.argmax(self.sess.run(self.layer[-1],feed_dict={self.xs:X}),axis=1)