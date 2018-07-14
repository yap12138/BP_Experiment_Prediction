# -*-coding:utf-8-*-
import tensorflow as tf
import docRead
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BPNN:
    def __init__(self, input_n, hidden_n, output_n):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.input_n = input_n
        self.output_n = output_n

        # 判断是否初始化变量
        self.has_init_var = False

        with self.graph.as_default():
            # 2.定义节点准备接收数据
            # define placeholder for inputs to network
            self.xs = tf.placeholder(tf.float32, [None, input_n], name="input-x")
            self.ys = tf.placeholder(tf.float32, [None, output_n], name="input-y")

            # 3.定义神经层：隐藏层和预测层
            # add hidden layer 输入值是 xs，在隐藏层有 14 个神经元
            l1 = self._add_layer(self.xs, input_n, hidden_n, activation_function=tf.nn.sigmoid,
                                 weight_name="hw1", biases_name="hb1")
            # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
            self.prediction = self._add_layer(l1, hidden_n, output_n, activation_function=None,
                                              weight_name="ow1", biases_name="ob1")

            # 4.定义 loss 表达式
            # the error between prediction and real data
            self.loss = tf.reduce_mean(tf.square(self.ys - self.prediction), name="loss")

            # 5.选择 optimizer 使 loss 达到最小
            # 这一行定义了用什么方式去减少 loss，学习率是 0.1
            self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.loss, name="train_step")

    # 添加层
    def _add_layer(self, inputs, in_size, out_size, activation_function=None, weight_name=None, biases_name=None):
        with self.graph.as_default():
            # add one more layer and return the output of this layer
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name=weight_name)
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name=biases_name)
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            return outputs

    # 一次前馈网络预测
    def predict(self, x_data):
        with self.graph.as_default():
            return self.sess.run(self.prediction, feed_dict={self.xs: x_data})

    # 输入训练数据和标签，训练模型
    def train(self, x_data, y_data, times=10000):
        if np.shape(x_data)[1] != self.input_n and np.shape(y_data)[1] != self.output_n:
            raise ValueError("训练数据维度不一致")

        dataset_set = np.shape(x_data)[0]
        batch_size = 8

        with self.graph.as_default():
            # important step 对所有变量进行初始化
            init_op = tf.global_variables_initializer()
            # 上面定义的都没有运算，直到 sess.run 才会开始运算
            self.sess.run(init_op)

            # 迭代 1000 次学习，sess.run optimizer
            for i in range(times):
                start = (i * batch_size) % dataset_set
                end = min(start+batch_size, dataset_set)
                # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
                self.sess.run(self.train_step, feed_dict={self.xs: x_data[start:end], self.ys: y_data[start:end]})
                if i % 2000 == 0:
                    # to see the step improvement
                    print(self.sess.run(self.loss, feed_dict={self.xs: x_data, self.ys: y_data}))

    # 保存当前BP网络的计算图
    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path+"/model.ckpt")

    # 从path中加载BP网络计算图，如果是当前目录需要加上前缀"./"
    def load(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path+"/model.ckpt")
            self.has_init_var = True


if __name__ == '__main__':
    data_path = r'F:\495_work\Workspaces-School\学期资料\大三资料\大三下\人工智能\综合性实验\人工智能实验评分样本'
    # 1.训练的数据
    x_input, y_input = docRead.getTrainData(data_path)

    nn = BPNN(8, 14, 1)
    # nn.train(x_input, y_input)
    nn.load("./BP_test")
    result = nn.predict(x_input)
    print("result:")
    print(result)
    # nn.save("BP_test")
