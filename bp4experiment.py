# -*-coding:utf-8-*-
import tensorflow as tf
import docRead
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def train(data_path):
    # 1.训练的数据
    # Make up some real data
    # x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # noise = np.random.normal(0, 0.05, np.shape(x_data))
    # y_data = [[0], [1], [1], [0]]
    x_data, y_data = docRead.getTrainData(data_path)

    dataset_set = np.shape(x_data)[0]
    batch_size = 8

    # 2.定义节点准备接收数据
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 8])
    ys = tf.placeholder(tf.float32, [None, 1])

    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
    l1 = add_layer(xs, 8, 14, activation_function=tf.nn.sigmoid)
    # l2 = add_layer(l1, 10, 15, activation_function=tf.nn.sigmoid)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = add_layer(l1, 14, 1, activation_function=None)

    # 4.定义 loss 表达式
    # the error between prediction and real data
    '''loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    '''
    loss = tf.reduce_mean(tf.square(ys - prediction))

    # 5.选择 optimizer 使 loss 达到最小
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    # train_step = tf.train.AdagradOptimizer(0.3).minimize(loss)

    # important step 对所有变量进行初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)

    # 迭代 1000 次学习，sess.run optimizer
    for i in range(10000):
        start = (i * batch_size) % dataset_set
        end = min(start+batch_size, dataset_set)
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        sess.run(train_step, feed_dict={xs: x_data[start:end], ys: y_data[start:end]})
        if i % 2000 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    rst = sess.run(prediction, feed_dict={xs: x_data})
    return rst


if __name__ == '__main__':
    dir = r'F:\495_work\Workspaces-School\学期资料\大三资料\大三下\人工智能\综合性实验\人工智能实验评分样本'
    result = train(dir)
    print("result:")
    print(result)
