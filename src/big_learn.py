import theano as th
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import scipy

# DECLARE INPUTS
n_in, n_out = 784, 10      # MNIST-sized
input, target = T.dvector(), T.iscalar()
W = th.shared(np.zeros((n_in, n_out)))
b = th.shared(np.zeros(n_out))
# DEFINE THE GRAPH
probs = nnet.softmax(T.dot(input, W) + b)
pred = T.argmax(probs)
nll = -T.log(probs)[T.arange(target.shape[0]), target]
#T.arange(target.shape[0]), target]
dW, db = T.grad(nll, [W, b])
# COMPILE THE GRAPH
test = th.function([input], pred)
train = th.function([input, target], nll,
updates = {W: W - 0.1*dW, b: b - 0.1*db})
