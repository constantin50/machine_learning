import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class TfLinreg(object):
  def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
    self.x_dim = x_dim;
    self.learning_rate=learning_rate;
    self.g = tf.Graph()

    with self.g.as_default():
      tf.set_random_seed(random_seed)

      self.build()

      self.init_op = tf.global_variables_initializer()
  
  def build(self):

    # placeholders for input data
    self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim),
                            name='x_input')
    
    self.y = tf.placeholder(dtype=tf.float32, shape=(None), name='y_input')

    print(self.X)
    print(self.y)

    # weight matrix and bias vector
    w = tf.Variable(tf.zeros(shape=(1)), name='weight')

    b = tf.Variable(tf.zeros(shape=(1)), name='bias')

    print(w)
    print(b)

    self.z_net = tf.squeeze(w*self.X + b, name='z_net') 

    print(self.z_net)
    
    # ordinary least squares cost function
    sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')

    print(sqr_errors)

    self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')

    # find extremum of cost function
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=self.learning_rate,
      name='GradientDescent')
    
    self.optimizer = optimizer.minimize(self.mean_cost)

# function for model training 
def train_linreg(sess, model, X_train, y_train, num_epochs=10):
  # initialize all variables: W and b
  sess.run(model.init_op)

  training_costs = []
  for i in range(num_epochs):
    _, cost = sess.run([model.optimizer, model.mean_cost], 
                       feed_dict={model.X:X_train, 
                                  model.y:y_train})
    training_costs.append(cost)
  
  return training_costs


# function for model training 
def train_linreg(sess, model, X_train, y_train, num_epochs=10):
  # initialize all variables: W and b
  sess.run(model.init_op)

  training_costs = []
  for i in range(num_epochs):
    _, cost = sess.run([model.optimizer, model.mean_cost], 
                       feed_dict={model.X:X_train, 
                                  model.y:y_train})
    training_costs.append(cost)
  
  return training_costs


def predict_linreg(sess, model, X_test):
  y_pred = sess.run(model.z_net, feed_dict={model.X:X_test})
  return y_pred
  
def plot_model(X_train, y_train, model):
  plt.scatter(X_train, y_train, marker='s', s=50, label='Training Data')
  plt.plot(range(X_train.shape[0]), predict_linreg(sess, lrmodel, X_train),
  color='gray', marker='o', markersize=6, linewidth=3, label='LinReg Model')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.tight_layout()
  plt.show()
  
  
# example:

X_train = np.arange(10).reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)
sess = tf.Session(graph=lrmodel.g)
