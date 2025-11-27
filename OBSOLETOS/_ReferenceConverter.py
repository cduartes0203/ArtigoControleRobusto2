import numpy as np
import pandas as pd


class ReferenceConverter:
    def __init__(self, RUL_ref=int(46.49*3600), D_max=1000):
        self.k = 1
        self.rul_ref = RUL_ref
        self.D_max = D_max
        self.beta_ref = 0
    
    def update(self, D_hat, prnt=False):
        self.rul_ref = self.rul_ref-10
        #print('rul ref:',self.rul_ref,'k:',self.k)
        self.beta_ref = (self.D_max-D_hat)/self.rul_ref
        self.k = self.k + 1

        return self.beta_ref
        
