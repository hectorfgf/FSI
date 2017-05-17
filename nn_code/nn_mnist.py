import gzip
import cPickle

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


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

print train_y.size
print valid_y.size
print test_y.size

print test_y[0]
test_y = one_hot(test_y,10)
print test_y[0]
print test_x[0].size
print test_y.size


valid_y = one_hot(valid_y,10)
train_y = one_hot(train_y,10)





# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt


plt.imshow(test_x[8].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[8], "aqui"


# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 45)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(45)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(45, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

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



batch_size = 100
errorAnterior = 7000
epoch = 0
entrenar = True
fin=0
errores=[] #hacer en la memoria una grafica de erres

while entrenar:
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    epoch = epoch + 1
    print "Epoch:", epoch, "Error:", error
    print "Error anterior: ", errorAnterior
    print "----------------------------------------------------------------------------------"
    print "validacion: ", epoch, "Error: ", sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    print "----------------------------------------------------------------------------------"
    if ( error >= errorAnterior):
        if (fin >= 10):
            entrenar=False
            break
        fin = fin+1
    else:
        fin=0
    errorAnterior = error
    errores.append(errorAnterior)

error = sess.run(loss, feed_dict={x: test_x, y_: test_y})
print "test: ", "Error: ", error
result = sess.run(y, feed_dict={x: test_x})
aciertos=0
for b, r in zip(test_y, result):
    if (np.argmax(b) == np.argmax(r)):
        aciertos = aciertos + 1
print "----------------------------------------------------------------------------------"
print "aciertos: ",aciertos," de ", test_y.size/10

plt.plot(errores)
plt.show()