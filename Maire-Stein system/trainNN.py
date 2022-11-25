## 算法 keras 自定义loss  Matlab data
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow_probability as tfp
import scipy.io as scio

# Data size on boundary condition of quasi-potential
Nu=400
dataFile = "xdata1"  
data = scio.loadmat(dataFile)
xdata1 = data['xdata1']
# xdata=tf.reshape(xdata, [Nu, 1])

dataFile = "xdata2"  
data = scio.loadmat(dataFile)
xdata2 = data['xdata2']
# ydata=tf.reshape(ydata, [Nu, 1])

dataFile = "xdata3"  
data = scio.loadmat(dataFile)
xdata3 = data['xdata3']
# Sphi=tf.reshape(Sphi, [Nu, 1])

dataFile = "xdata4"  
data = scio.loadmat(dataFile)
xdata4 = data['xdata4']

dataFile = "xdata5"  
data = scio.loadmat(dataFile)
xdata5 = data['xdata5']

# Data size on Hamiltionian
NH=5000
# DeepNN topology
layers=[2,20,20,20,20,3]
# layers=[2,50,50,20,1]
tf_optimizer = tf.keras.optimizers.Adam(
   learning_rate=0.05,
   beta_1=0.99,
   epsilon=1e-1)
# tf_optimizer = tf.keras.optimizers.SGD

# Jacobi matrix
Jacob=tf.constant([[-2.0,0],[0,-2]])
M=tf.constant([[4.0,0],[0,4]],dtype=tf.float32)
xnode=tf.constant([-1.0,0],dtype=tf.float32)
delta=0.1

xmax=0
xmin=-1.5
ymax=0.6
ymin=-0.6

#### ----------Action plot boundary condition，正式training------------
xdata1 =tf.reduce_sum(xdata1, axis=0)
xdata2 =tf.reduce_sum(xdata2, axis=0)
X_u_train = tf.stack([xdata1,xdata2], axis=1)
xdata3 =tf.reduce_sum(xdata3, axis=0)
xdata4 =tf.reduce_sum(xdata4, axis=0)
xdata5 =tf.reduce_sum(xdata5, axis=0)
u_train=tf.stack([xdata3,xdata4,xdata5], axis=1)
u_train=tf.cast(u_train, dtype=tf.float32)
X_u_train=tf.cast(X_u_train, dtype=tf.float32)
# print(X_u_train)
#### ------------------------------------------------------------------

# sampling points
x=tf.random.uniform([NH],xmin,xmax,dtype=tf.float32)
y=tf.random.uniform([NH],ymin,ymax,dtype=tf.float32)

# # 数据集乱序
# np.random.seed(116)
# np.random.shuffle(x_train)
# np.random.seed(116)
# np.random.shuffle(y_train)
# tf.random.set_seed(116)

class mymodel():
  def __init__(self,layers,optimizer,x,y):
    self.u_model=tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(
      lambda X: X))
    for width in layers[1:5]:
      self.u_model.add(tf.keras.layers.Dense(
        width, activation=tf.nn.tanh,
        kernel_initializer='glorot_normal'))
        
    self.u_model.add(tf.keras.layers.Dense(
      layers[5], 
      kernel_initializer='glorot_normal'))
        
    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))
        
    self.optimizer=optimizer
    self.dtype=tf.float32

  # Defining custom loss
  def __loss(self, u, u_pred, u_pred2):
    H_pred = self.H_model()
    return tf.reduce_mean(tf.square(u[:,0] - u_pred[:,0]),axis=0)  + \
     tf.reduce_mean(tf.square(u[:,1] - u_pred[:,1]),axis=0)  + \
     tf.reduce_mean(tf.square(u[:,2] - u_pred[:,2]),axis=0)  + \
     tf.reduce_mean(tf.square(u_pred2[:,0]),axis=0) * 1 + \
     tf.reduce_mean(tf.square(u_pred2[:,1]),axis=0) * 1 + \
     tf.reduce_mean(tf.square(u_pred2[:,2]),axis=0) * 1 + \
     H_pred * 1
  
  def __grad(self, X, u):
    with tf.GradientTape() as tape:
      Y=self.u_model(X)
      Xnode=tf.constant([-1,0],shape=(1,2),dtype=tf.float32)
      Y2=self.u_model(Xnode)
      loss_value = self.__loss(u, Y, Y2)
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    return var
  
  # The actual PINN
  def H_model(self):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching the two inputs we’ll need later, x and t
      tape.watch(x)
      tape.watch(y)
      # Packing together the inputs
      X_f = tf.stack([x,y], axis=1)

      # Getting the prediction
      u = self.u_model(X_f)
      u0=u[:,2]
    
    # Getting the derivative
    u_x = tape.gradient(u0, x)
    u_y = tape.gradient(u0, y)

    # Letting the tape go
    del tape

    gamma=1
    b1=x-tf.pow(x,3)-gamma*tf.multiply(x,tf.pow(y,2))
    b2=-tf.multiply((1+tf.pow(x,2)),y)
    
    L1=tf.reduce_mean(tf.square(u_x-u[:,0]),axis=0)
    L2=tf.reduce_mean(tf.square(u_y-u[:,1]),axis=0)
    L3=tf.reduce_mean(tf.square(b1*u[:,0]+b2*u[:,1]+0.5*(tf.pow(u[:,0],2)+tf.pow(u[:,1],2))),axis=0)

    # Buidling the Hamiltonian
    return L1+L2+L3

  # def get_weights(self):
  #   w = []
  #   for layer in self.u_model.layers[1:]:
  #     weights_biases = layer.get_weights()
  #     weights = weights_biases[0].flatten()
  #     biases = weights_biases[1]
  #     w.extend(weights)
  #     w.extend(biases)
  #   return tf.convert_to_tensor(w, dtype=self.dtype)

  # def set_weights(self, w):
  #   for i, layer in enumerate(self.u_model.layers[1:]):
  #     start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
  #     end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
  #     weights = w[start_weights:end_weights]
  #     w_div = int(self.sizes_w[i] / self.sizes_b[i])
  #     weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
  #     biases = w[end_weights:end_weights + self.sizes_b[i]]
  #     weights_biases = [weights, biases]
  #     layer.set_weights(weights_biases)

  def summary(self):
    return self.u_model.summary()

  # The training function
  def fit(self, X_u, u, tf_epochs=5000):
    # Creating the tensors
    #X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    #u = tf.convert_to_tensor(u, dtype=self.dtype)
    #print(self.__wrap_training_variables())

    LOSS=np.zeros([1,tf_epochs])
    for epoch in range(tf_epochs):
      # Optimization step
      loss_value, grads = self.__grad(X_u, u)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
      LOSS[0,epoch]=loss_value.numpy()
      print('epoch, loss_value:', epoch, loss_value)

    scio.savemat('LOSS.mat',{'LOSS':LOSS})
    # def loss_and_flat_grad(w):
    #   with tf.GradientTape() as tape:
    #     self.set_weights(w)
    #     Xnode=tf.constant([-1,0],shape=(1,2),dtype=tf.float32)
    #     Y2=tf.reduce_sum(self.u_model(Xnode),axis=1)
    #     loss_value = self.__loss(u, self.u_model(X_u), Y2)
    #   grad = tape.gradient(loss_value, self.u_model.trainable_variables)
    #   grad_flat = []
    #   for g in grad:
    #     grad_flat.append(tf.reshape(g, [-1]))
    #   grad_flat =  tf.concat(grad_flat, 0)
    #   return loss_value, grad_flat

    # tfp.optimizer.lbfgs_minimize(
    #   loss_and_flat_grad,
    #   initial_position=self.get_weights(),
    #   num_correction_pairs=50,
    #   max_iterations=2000)
    # lbfgs(loss_and_flat_grad,
    #   self.get_weights())
    # lbfgs(loss_and_flat_grad,
    #   self.get_weights(),
    #   nt_config, Struct(), True,
    #   lambda epoch, loss, is_iter:
    #     self.logger.log_train_epoch(epoch, loss, "", is_iter))

    # print(self.__wrap_training_variables())

  def predict(self, X_star):
    u_star = self.u_model(X_star)
    # f_star = self.H_model()
    return u_star#, f_star


model=mymodel(layers, tf_optimizer, x, y)

# checkpoint_save_path="./checkpoint/mnist.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#   print('-------------------load the model---------------------')
#   model.load_weights(checkpoint_save_path)

# cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                         save_weights_only=True,
#                         save_best_only=True)

history=model.fit(X_u_train, u_train, tf_epochs=50000)

model.summary()

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
x1=tf.constant([-1.5,-1.5,-1.5,-1,-1,-1,0,0,0],dtype=tf.float32)
y1=tf.constant([-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6],dtype=tf.float32)
X_star = tf.stack([x1,y1], axis=1)
print('X_star:\n ', X_star)
u_pred= model.predict(X_star)
print('u_pred:\n ', u_pred)

dataFile = "xtest1"  
data = scio.loadmat(dataFile)
xtest1 = data['xtest1']
dataFile = "xtest2"  
data = scio.loadmat(dataFile)
xtest2 = data['xtest2']
xtest1 =tf.reduce_sum(xtest1, axis=0)
xtest2 =tf.reduce_sum(xtest2, axis=0)
X_star = tf.stack([xtest1,xtest2], axis=1)
Stest= model.predict(X_star)
scio.savemat('Stest.mat',{'Stest':Stest.numpy()})
