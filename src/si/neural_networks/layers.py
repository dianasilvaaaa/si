from abc import ABCMeta, abstractmethod

class Layer (metaclass = ABCMeta):

    @abstractmethod
    def foward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, input):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self._class_._name_


import numpy as np    

class DenseLayer(Layer):

    def __init__(self, n_units, input_shape = None):
        super().__init__()
        self.n_units = n_units
        self.input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None

    

    






