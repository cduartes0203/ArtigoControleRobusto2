import numpy as np
import pandas as pd

def Cp_calc(LAMBDA=0, PHI=0):
    path = r'Datasets/Cp_X_lambda.csv'
    df = pd.read_csv(path)
    dist = np.abs(df['lambda'].values - LAMBDA)

    return (df['cp'].values[np.argmin(dist)])

class Aerodynamics:
    def __init__(self, **params):

        self.Area =params['Area']
        self.R =params['R']
        self.RHO =params['RHO']
        self.PHI_opt = params['PHI_opt']

    def TAU_r(self, OMEGA_r, v, PHI=None):
        
        Area = self.Area
        R = self.R
        RHO = self.RHO
        if PHI == None:
            PHI = self.PHI_opt
        
        LAMBDA = OMEGA_r*R/v
        C_p = Cp_calc(LAMBDA)
        TAU = RHO*Area*C_p*(v**3)/(2*OMEGA_r)
        return TAU