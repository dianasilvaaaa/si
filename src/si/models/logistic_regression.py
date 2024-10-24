import numpy as np

from si.data.dataset import dataset

def LogisticRegression(l2_penalty = 0.01, alpha, max_inter, patience, scale):

    self.l2_penalty = l2_penalty 
    self.alpha = alpha
    self.max_inter = max_inter
    self.patience = patience
    self.scale = scale


    self.theta = None
    self.theta_zero = None
    self.mean = None
    self.std = None
    self.cost_history = {}

#def cost(self, dataset: dataset) -> float:
    



    
