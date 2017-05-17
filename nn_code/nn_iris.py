import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:105, 0:4].astype('f4')
x_validacion = data[105:127, 0:4].astype('f4')
x_test = data[127:, 0:4].astype('f4')
# the samples are the four first rows of data
y_data = one_hot(data[:105, 4].astype(int), 3)# the labels are in the last row. Then we encode them in one hot code
y_validacion = one_hot(data[105:127, 4].astype(int), 3)
y_test = one_hot(data[127:, 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
print "----------------------"
print "   Start training...  "
print "----------------------"
batch_size = 20
errorAnterior = 10000000
epoch = 0
entrenar = True
fin=False
errores=[]
while entrenar:
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    error = sess.run(loss, feed_dict={x: x_validacion, y_: y_validacion})
    epoch=epoch+1;
    print "Epoch #:", epoch, "Error: ", error
    print "Error anterior: ", errorAnterior
    """result = sess.run(y, feed_dict={x: batch_xs})
    aciertos=0
    for b, r in zip(batch_ys, result):
        valid = 0
        if (np.argmax(b) == np.argmax(r)):
            aciertos = aciertos + 1
            valid = 1
        print b, "-->", r, "--> ", valid
    print "----------------------------------------------------------------------------------"
    print aciertos
    print "----------------------------------------------------------------------------------
    print "validacion: ", epoch, "Error: ", sess.run(loss, feed_dict={x: x_validacion, y_: y_validacion})
    result = sess.run(y, feed_dict={x: x_validacion})
    aciertos = 0
    for b, r in zip(y_validacion, result):
        valid = 0
        if (np.argmax(b) == np.argmax(r)):
            aciertos = aciertos + 1
            valid = 1
        print b, "-->", r, "--> ", valid
    print "----------------------------------------------------------------------------------"
    print aciertos
    print "----------------------------------------------------------------------------------"""""
    if(error>=errorAnterior):
        if (fin):
            entrenar = False
            break
        fin = True
    else:
        fin = False
    errorAnterior=error
    errores.append(errorAnterior)
print "test: ", "Error: ", sess.run(loss, feed_dict={x: x_test, y_: y_test})
result = sess.run(y, feed_dict={x: x_test})
aciertos = 0
for b, r in zip(y_test, result):
    valid=0
    if (np.argmax(b) == np.argmax(r)):
        aciertos = aciertos+1
        valid = 1
    print b, "-->", r, "--> ", valid
print "----------------------------------------------------------------------------------"
print "aciertos: ",aciertos," de ", y_test.size/3

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.plot(errores)
plt.show()