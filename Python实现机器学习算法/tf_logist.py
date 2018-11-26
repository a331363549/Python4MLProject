import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
# from sklearn.cros import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

# 特征shape
numFeatures = trainX.shape[1]
# 标签shape
numLabels = trainY.shape[1]
# 占位符
X = tf.placeholder(tf.float32, [None, numFeatures])
y_ = tf.placeholder(tf.float32, [None, numLabels])
# W b
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

# 随机抽取标准偏差为0.01的正态分布
weights = tf.Variable(tf.random_normal([numFeatures, numLabels], mean=0.0, stddev=0.01, name="weights"))
bias = tf.Variable(tf.random_normal([1, numLabels], mean=0, stddev=0.01, name="bias"))

# Logistic回归方程三要素
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

# Epoch 训练次数和学习率
numEpochs = 700
# learninng_rate
learningRate = tf.train.exponential_decay(learning_rate=0.0008, global_step=1, decay_steps=trainX.shape[0], decay_rate=0.95,
                                          staircase=True)
# 损失函数
cost_OP = tf.nn.l2_loss(activation_OP - y_, name="squared_error_cost")
# 定义渐变下降
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)



# 创建会话，初始化变量
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 设置变量追踪训练过程
# tf.argmax(activation_OP, 1)  以最大概率返回标签
#  tf.argmax(y_, 1) 正确的标签
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(y_, 1))
# 如果每个错误的预测为0且每个真是预测为1，则平均值会返回我们的准确性
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# 汇总op的回归输出
activation_summary_OP = tf.summary.histogram("output", activation_OP)
# 汇总的准确度
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# 汇总OP的成本
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# 汇总检查每次迭代后变量W b是如何更新
weightSummary = tf.summary.histogram('weights', weights.eval(session=sess))
biasSummary = tf.summary.histogram('biases', bias.eval(session=sess))
# 合并所有的汇总
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])
# 汇总writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

# trianing
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []
# 训练epochs
for i in range(numEpochs):
    if i > 1 and diff < 0.0001:
        print("change in cost %g; convergence." % diff)
        break
    else:
        step = sess.run(training_OP, feed_dict={X: trainX, y_: trainY})
        if i % 100 == 0:
            # 将epoch添加到epoch_values
            epoch_values.append(i)
            # 基于测试集数据生成准确度
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, y_: trainY})
            # 为实时图形变量添加准确性
            accuracy_values.append(train_accuracy)
            # 为实时图形变量添加成本
            cost_values.append(newCost)
            # 对变量值重新分配值
            diff = abs(newCost - cost)
            cost = newCost
            # 生成输出语句
            print("step %d, training accuracy %g, cost %g, change in cost %g" % (i, train_accuracy, newCost, diff))

# 预测
print("final accuracy on test set : %s " % str(sess.run(accuracy_OP, feed_dict={X: testX, y_: testY})))
