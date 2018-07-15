# -*-coding:utf-8-*-
import tensorflow as tf
import docRead
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 基于BP网络模型的3层BP模型
class BPNN:
    # 三个参数分别为输入层，隐含层，输出层节点个数
    def __init__(self, input_n, hidden_n, output_n):
        self.__graph = tf.Graph()
        self.__sess = tf.Session(graph=self.__graph)

        self.__input_n = input_n
        self.__output_n = output_n

        self.__has_load = False

        self.__loss_threshold = 5.0

        with self.__graph.as_default():
            # 1.定义节点准备接收数据
            # define placeholder for inputs to network
            self.__xs = tf.placeholder(tf.float32, [None, input_n], name="input-x")
            self.__ys = tf.placeholder(tf.float32, [None, output_n], name="input-y")

            # 2.定义神经层：隐藏层和预测层
            # add hidden layer 输入值是 xs，在隐藏层有 hidden_n 个神经元， （本实验最好为14个）
            l1 = self.__add_layer(self.__xs, input_n, hidden_n, activation_function=tf.nn.sigmoid,
                                  weight_name="hw1", biases_name="hb1")
            # add output layer 输入值是隐藏层 l1，在预测层输出 output_n 个结果，  （本实验为1个）
            self.__prediction = self.__add_layer(l1, hidden_n, output_n, activation_function=None,
                                                 weight_name="ow1", biases_name="ob1")

            # 3.定义 loss 表达式
            # the error between prediction and real data
            self.__loss = tf.reduce_mean(tf.square(self.__ys - self.__prediction), name="loss")

            # 4.选择 optimizer 使 loss 达到最小
            # 这一行定义了用什么方式去减少 loss，学习率是 0.1
            self.__train_step = tf.train.AdamOptimizer(0.1).minimize(self.__loss, name="train_step")

    def __del__(self):
        self.__sess.close()

    # 添加层
    def __add_layer(self, inputs, in_size, out_size, activation_function=None, weight_name=None, biases_name=None):
        with self.__graph.as_default():
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
        with self.__graph.as_default():
            return self.__sess.run(self.__prediction, feed_dict={self.__xs: x_data})

    # 输入训练数据和标签，训练模型
    def train(self, x_data, y_data, times=10000):
        if np.shape(x_data)[1] != self.__input_n and np.shape(y_data)[1] != self.__output_n:
            raise ValueError("训练数据维度不一致")

        dataset_set = np.shape(x_data)[0]
        batch_size = 8

        with self.__graph.as_default():
            # 如果从文件入的模型则跳过初始化步骤
            if not self.__has_load:
                # important step 对所有变量进行初始化
                init_op = tf.global_variables_initializer()
                # 上面定义的都没有运算，直到 sess.run 才会开始运算
                self.__sess.run(init_op)

            # 迭代 times 次学习，sess.run optimizer
            for i in range(times):
                start = (i * batch_size) % dataset_set
                end = min(start+batch_size, dataset_set)
                # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
                self.__sess.run(self.__train_step, feed_dict={self.__xs: x_data[start:end], self.__ys: y_data[start:end]})
                if i % 2000 == 0:
                    # to see the step improvement
                    loss_cur = self.__sess.run(self.__loss, feed_dict={self.__xs: x_data, self.__ys: y_data})
                    print("at %d times, error is %f" % (i, loss_cur))
                    if loss_cur < self.__loss_threshold:
                        break
            loss_cur = self.__sess.run(self.__loss, feed_dict={self.__xs: x_data, self.__ys: y_data})
            print("at last, error is %f" % loss_cur)

    # 保存当前BP网络的计算图
    def save(self, path):
        with self.__graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.__sess, path + "/model.ckpt")

    # 从path中加载BP网络计算图，如果是当前目录需要加上前缀"./"
    def load(self, path):
        with self.__graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.__sess, path + "/model.ckpt")
            self.__has_load = True


if __name__ == '__main__':
    data_path = r'F:\495_work\Workspaces-School\学期资料\大三资料\大三下\人工智能\综合性实验\人工智能实验评分样本'
    # 训练的数据
    x_input, y_input = docRead.getTrainData(data_path)

    nn = BPNN(8, 14, 1)
    # nn.load("./BP_test")
    nn.train(x_input[0:180], y_input[0:180])
    result = nn.predict(x_input[181:240])
    print("result:")
    print(result)
    # nn.save("BP_test")
