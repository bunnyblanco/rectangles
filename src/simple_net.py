import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import nnet

W_in = T.dmatrix('W_in')
B_in = T.dvector('B_in')
X_in = T.dvector('X_in')
A_h = T.dvector('A_h')
A_h = T.tanh(T.dot(W_in,X_in) + B_in)
L_h = th.function([W_in, B_in, X_in], A_h)

X_h = T.vector('X_h')
W_h = T.matrix('W_h')
B_h = T.vector('B_h')
A_out = T.dvector('A_out')
A_out = nnet.softmax(T.dot(W_h, X_h) + B_h)
L_out = th.function([W_h, B_h, X_h], A_out)


if __name__=='__main__':
    n_h = 1000
    f = open('./process/rectangles.csv','r')
    first = True
    for line in f:
        if first:
            x_data = np.array(line.split(',')[1:]) #Kludge to get rid of a space
            first = False
        else:
            x_data = np.vstack((x_data, line.split(',')[1:])) #See above

    N_s, n_in = x_data[:,:-1].shape #The last element is the output

    x_in = np.zeros((n_in), dtype='float')
    w_in = np.random.rand(n_h, n_in)
    b_in = np.zeros((n_h), dtype='float')

    w_h = np.random.rand(1, n_h)
    b_h = np.zeros((1), dtype='float')

    print("running "+str(N_s)+" samples...") #Need to run a subset and then validate
    for n in range(0, N_s):
        x_n = x_data[n,:-1]
        y_0 = x_data[n,-1]
        i = 0
        for x_0 in x_n:
            x_in[i] = float(x_0)
            i += 1

        l_h = L_h(w_in, b_in, x_in)
        l_out = L_out(w_h, b_h, l_h)
    #    print(y_0 - l_out)
