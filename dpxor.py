# -*-coding:utf-8-*-
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BPNN:
    def __init__(self, ni, nh, no):
        # 下面三个变量保存各层神经元个数
        self.input_n = ni
        self.hidden_n = nh
        self.output_n = no
        # 下面两个变量保存权重
        self.input_weights = tf.Variable(tf.random_normal([ni, nh], stddev=1, seed=1))
        self.output_weights = tf.Variable(tf.random_normal([nh, no], stddev=1, seed=1))
        # 输入值和期望输出值
        self.input_cells = tf.placeholder(tf.float32, shape=(None, ni), name="x-input")
        self.mean_cells = tf.placeholder(tf.float32, shape=(None, no), name="y-input")
        # 输入层和隐含层的偏置值
        input_biases = tf.Variable(tf.zeros([1, nh]) + 1)
        hidden_biases = tf.Variable(tf.zeros([1, no]) + 1)

        self.hidden_cells = tf.nn.sigmoid(tf.matmul(self.input_cells, self.input_weights) + input_biases)
        output_tmp = tf.matmul(self.hidden_cells, self.output_weights) + hidden_biases

        self.output_cells = output_tmp

        # self.cross_entropy = tf.reduce_mean(0.5 * tf.pow(self.mean_cells - self.output_cells, 2))
        '''self.cross_entropy = -tf.reduce_mean(
            self.mean_cells * tf.log(tf.clip_by_value(self.output_cells, 1e-10, 1.0)) +
            (1-self.mean_cells) * tf.log(tf.clip_by_value((1-self.output_cells), 1e-10, 1.0))
        )'''
        self.cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.mean_cells - self.output_cells)))
        # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.mean_cells, logits=self.output_cells)
        self.train_step = tf.train.AdadeltaOptimizer(0.1).minimize(self.cross_entropy)

        self.sess = tf.Session()

    def __del__(self):
        self.sess.close()

    def predict(self, cases):
        # sess = tf.Session()
        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)
        return self.sess.run(self.output_cells, feed_dict={self.input_cells: cases[:]})

    def train(self, cases, labels, limit=10000):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        for i in range(limit):
            self.sess.run(self.train_step, feed_dict={self.input_cells: cases[:], self.mean_cells: labels[:]})
            if i % 2000 == 0:
                total_cross_entropy = self.sess.run(self.cross_entropy, feed_dict={self.input_cells: cases[:],
                                                                                   self.mean_cells: labels[:]})
                print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
                # print("After ", i, "training step(s), cross entropy on all data is ", total_cross_entropy)

    def test(self):
        # 训练数据
        cases = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
        labels = [[0], [1], [1], [0]]
        self.train(cases, labels)

        print(self.predict(cases))


if __name__ == '__main__':
    nn = BPNN(2, 5, 1)
    nn.test()
