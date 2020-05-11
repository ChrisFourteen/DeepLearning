
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res

dataset_meta = unpickle('cifar-100-python/meta')
train_set = unpickle('cifar-100-python/train')
X_train = train_set[b'data']
y_train = train_set[b'coarse_labels']
X_train = X_train[0:30000].reshape(-1, 3, 32, 32)
X_train = np.rollaxis(X_train, 1, 4)
y_train = (np.arange(20) == np.array(y_train[0:30000]).reshape(-1, 1)).astype(np.int)
X_train_FIN = []

# 3通道图像训练时，总显示现存不足，转换为灰度图像
for i in range(len(X_train)):
    gray = cv2.cvtColor(X_train[i],cv2.COLOR_RGB2GRAY)
    X_train_FIN.append(gray / 255)

X_train_FIN = np.array(X_train_FIN)
X_train_FIN = np.expand_dims(X_train_FIN,3)
print(X_train_FIN.shape)

class network:
    def __init__(self, X_true, y_true, n_classes, learning_rate):
        self.graph = tf.Graph()
        self.dir = './CIFAR100_RES'
        with self.graph.as_default():
            self.classes = n_classes
            self.X = X_true
            self.y = y_true
            self.leaning_rate = learning_rate
            self.X_input = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
            self.y_input = tf.placeholder(tf.float32, shape=(None, n_classes))

            # 输入 32*32*3,卷积核大小 5*5，步长1，填充方式valid  ---> 输出 28*28*100
            # 输出计算公式： out_width = (input_width - kernel_size + 1)/strides
            conv1 = tf.layers.conv2d(self.X_input, filters=50, kernel_size=5, strides=1, padding='VALID')

            # 池化输出结果 14*14*100
            self.conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

            # 输入 14*14*100，卷积核大小 3*3，步长1，填充方式valid  ---> 输出 12*12*150
            conv2 = tf.layers.conv2d(self.conv1, filters=100, kernel_size=3, strides=1, padding='VALID')

            # 池化输出结果 6*6*150
            self.conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

            # 输入 6*6*150，卷积核大小 3*3，步长1，填充方式 SAME  ---> 输出 6*6*250
            conv3 = tf.layers.conv2d(self.conv2, filters=120, kernel_size=3, strides=1, padding='SAME')

            # 池化输出结果 3*3*120
            self.conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)
            # 将卷积结果拉平，维度为 3*3*120
            self.f0 = tf.layers.flatten(self.conv3)

            self.f1 = tf.layers.dense(self.f0, 512, activation=tf.nn.relu)

            self.f2 = tf.layers.dense(self.f1, 150, activation=tf.nn.relu)

            self.out = tf.layers.dense(self.f2, self.classes)
            # 计算交叉熵
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.out,
                    labels=self.y_input)
                )
            # 训练
            with tf.name_scope('train'):
                self.opt = tf.train.AdamOptimizer(self.leaning_rate).minimize(self.loss)
            # 计算准确率
            with tf.name_scope('acc'):
                acc_pred = tf.equal(tf.argmax(self.out, axis=1), tf.argmax(self.y_input, axis=1))
                self.acc = tf.reduce_mean(tf.cast(acc_pred, tf.float32))
            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            if not os.path.isdir(self.dir):
                os.mkdir(self.dir)
            self.ckpt = tf.train.get_checkpoint_state(self.dir)
            if self.ckpt and self.ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.dir))
                print('Successfully loaded:', tf.train.latest_checkpoint(self.dir))
            else:
                print('Can not find old network weight')

    def train(self, x, y):
        feed_dict = {
            self.X_input: x,
            self.y_input: y
        }

        _c, l, a = self.sess.run([self.opt, self.loss, self.acc], feed_dict=feed_dict)
        return l, a

    def save(self):
        save_path = self.saver.save(self.sess, os.path.join(self.dir, 'best_model.ckpt'))
        print('model saved in {}'.format(save_path))

learning_rate = 1e-4
Epoch_total = 5
batch_size = 10
n_classes = 20

def next_batch(score, batch_size):
    batch_num = len(score) // batch_size
    for i in range(batch_num):
        yield score[i:((i + 1) * batch_size)]

net = network(X_train_FIN, y_train, n_classes, learning_rate)
for epoch in range(Epoch_total):
    loss = 0
    acc = 0
    X = next_batch(X_train_FIN, batch_size)
    Y = next_batch(y_train, batch_size)
    for i in range(len(net.X) // batch_size):
        cur_loss, cur_acc = net.train(next(X), next(Y))
        loss += cur_loss
        acc += cur_acc

        if i % 20 == 0:
            print('Epoch {} , batch: {} / {} , loss: {} ,  acc : {}'.format(epoch + 1, i, len(net.X) // batch_size,
                                                                            cur_loss, cur_acc))
    print('LOSS : {} , ACCURACY: {}'.format(loss / (len(net.X) // batch_size), acc / (len(net.X) // batch_size)))
net.save()


