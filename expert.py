from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start interactive session
sess = tf.InteractiveSession()

# build a softmax regression model with a single linear layer
## define placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
## define variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
## initialize variables
sess.run(tf.initialize_all_variables())

# implement regression model
## matmul -> matrix multiplication
## x = vectorized image
## W = weight matrix 
## b = bias
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define cost function
## reduce_sum sums across all classes
## reduce_mean takes average over the sums
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train the model
## add new operations to the computation graph: compute gradients, cpmpute parameter update steps, apply update steps to parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
## when run, train_step will apply the gradient descent updates to the parameters
## training the model is accomplished by repeating train_step
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})
## each iteration will load 50 training examples


# evaluate the model
## compare predictions with actual labels
## returns list of booleans
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
## check which fraction is correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


