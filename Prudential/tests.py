import numpy as np
import unittest

class TestNeuralMethods(unittest.TestCase):
    def test_gradient_descent(self):
        a = np.matrix('1 2; 3 4')