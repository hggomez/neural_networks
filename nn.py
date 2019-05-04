import struct
import numpy as np
from itertools import chain
from scipy.special import expit

def sigmoid(x):
  return expit(x)

def sigmoid_prime(x):
  return sigmoid(x) * (1-sigmoid(x))

class NN(object):

  def __init__(self, sizes, activation_func, activation_func_derivative, debug=False):
    #Uniform(0,1) distribution for weights and biases    
    #self.weights = [np.random.uniform(size=(x,y)) for x,y in zip(sizes[1:], sizes[:-1])]
    #self.biases = [np.random.uniform(size=(x,1)) for x in sizes[1:]]
    #Normal(0,1) distribution for weights and
    self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
    self.biases = [np.random.randn(x,1) for x in sizes[1:]]
    #Ones for the weights, zeros for the biases
    #self.weights = [np.ones((x,y)) for x,y in zip(sizes[1:], sizes[:-1])]
    #self.biases = [np.zeros((x,1)) for x in sizes[1:]]
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative

  def feedforward(self, sample):
    activations = [sample]
    weighted_sums = []
    for w, b in zip(self.weights, self.biases):
      weighted_sums.append((w @ activations[-1]) + b)
      activations.append(self.activation_func(weighted_sums[-1]))
    return weighted_sums, activations 

  def update_weights(self, samples, learning_rate):
    w, b = self.backpropagation(samples[0])
    for s in samples[1:]:
      w2, b2 = self.backpropagation(s)
      w = [wi + w2i for wi, w2i in zip(w, w2)]
      b = [bi + b2i for bi, b2i in zip(b, b2)]
    w = [wi / len(samples) for wi in w]
    b = [bi / len(samples) for bi in b]
    self.biases = [bi - np.transpose(b2i * learning_rate) for bi, b2i in zip(self.biases, b)]
    self.weights = [wi - np.transpose(w2i * learning_rate) for wi, w2i in zip(self.weights, w)]

  def backpropagation(self, sample):
    x, y = sample
    b_derivatives = [None] * len(self.biases)
    w_derivatives = [None] * len(self.weights)
    weighted_sums, activations = self.feedforward(x)
    b_derivatives[-1] = np.transpose(self.activation_func_derivative(weighted_sums[-1]) * (activations[-1] - y))
    w_derivatives[-1] = activations[-2] @ b_derivatives[-1]
    for i in reversed(range(len(self.weights)-1)):
      b_derivatives[i] = b_derivatives[i+1] @ self.weights[i+1] * np.transpose(self.activation_func_derivative(weighted_sums[i]))
      w_derivatives[i] = activations[i] @ b_derivatives[i]
    return w_derivatives, b_derivatives

  def mse(self, samples):
    squared_error = 0
    for x, y in samples:
      _, activations = self.feedforward(x)
      squared_error = squared_error + np.sum(np.square(activations[-1]-y))
    mse = squared_error / (2*len(samples))
    return mse
  
  def accuracy(self, samples):
    for x, y in samples: 
      pass
  
  def enable_debug(self):
    self.debug = True
  def disable_debug(self):
    self.debug = False


def load_data(images_file_path, labels_file_path):
  with open(images_file_path, 'rb') as images_file:
    images = parse_images(images_file)
  with open(labels_file_path, 'rb') as labels_file:
    labels = parse_labels(labels_file)
  return zip(images, labels)

def parse_images(images_file):
  _, amount, rows, columns = struct.unpack(">IIII", images_file.read(16))
  pixels = list(chain.from_iterable(struct.iter_unpack(">B", images_file.read())))
  image_size = rows*columns
  images = [np.array(pixels[i:i+image_size]).reshape(image_size, 1) for i in range(0, amount*image_size, image_size)]
  images = np.array(images)
  images = images / 255
  return images

def parse_labels(labels_file):
  _, amount = struct.unpack(">II", labels_file.read(8))
  labels = list(chain.from_iterable(struct.iter_unpack(">B", labels_file.read())))
  labels_vectors = np.zeros((amount, 10, 1))
  for l, l_v in zip(labels, labels_vectors):
    l_v[l] = 1.0
    l_v.reshape((10,1))
  return labels_vectors


#images = [np.array([0]*783 + [1]).reshape(784,1) for i in range(100)]
#labels = [np.array([0]*9 + [1]).reshape(10,1) for i in range(100)]
#data = list(zip(images, labels))

#images = [np.array([1,0]).reshape(2,1),\
#          np.array([1,1]).reshape(2,1),\
#          np.array([0,1]).reshape(2,1),\
#          np.array([0,0]).reshape(2,1)]
#labels = [np.array([1]).reshape(1,1),\
#          np.array([0]).reshape(1,1),\
#          np.array([1]).reshape(1,1),\
#          np.array([0]).reshape(1,1)]
#data = list(zip(images,labels))


nn = NN([784,16,16,10], sigmoid, sigmoid_prime, debug=False)
data = list(load_data("./dataset/train_images", "./dataset/train_labels"))
epochs = 1000
for i in range(epochs):
  nn.update_weights(data[:1], 0.05)
  if (i == 0 or i == epochs-1):
    print("EPOCH N°: ", i)
    print("Error", nn.mse(data))