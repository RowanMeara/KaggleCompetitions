import numpy as np
import math

class NeuralNet(object):
    def __init__(self,transfer_function, depth, breadth):
        self.transfer_function = NeuralNet.get_tf(transfer_function);

    @staticmethod
    def get_tf(tf):
        if tf == 'tanh':
            def trans(x):
                return np.tanh(x)
        # Placeholder until I get further along
        else tf == 'rbf':
            def trans(x):
                return np.tanh(x)
        return trans

    def forward_prop(self):
        return 0

    def back_prop(selfs):
        return 0

    @staticmethod
    def compute_gradient (w, x, y):
        """
        Computes the gradient using ridge regression
        Args:
            w : dx1 weight vector
            x : nxd input vector
            y : nx1 label vector
        Out:
            g : 1xd gradient of w
        """
        (n,d) = x.shape
        g = np.zeros(d)
        for i in range(0,d):
            g[i] = (w*x-y)*np.transpose(x[i])
        g += 0.5*w
        return g

    def grad_descent(self, x, y, max_error, max_iterations):
        """
        Args:
            x : nxd numpy matrix
            y : nx1 labels for each of the training vectors
            max_error : target precision
            max_iterations : maximum number of gradient descent iterations

        Out:
            w : dx1 post gradient descent weight vector
        """
        count = 0
        error = math.inf
        (n,d) = x.shape
        w = np.ones(n)
        s = 0.05
        while count < max_iterations and error > max_error:
            count += 1
            w_old = w
            # calculate new w
            w = w_old - s*self.compute_gradient(w_old,x,y)
            old_error = error
            t = (w - w_old)
            error = np.dot(t,t)
            # undo the last step if it increases error
            if error < old_error:
                s *= 1.01
            else:
                w = w_old
                s /= 2
        return w
