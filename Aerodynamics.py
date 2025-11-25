import numpy as np
import pandas as pd

def Cp_calc(LAMBDA=0, PHI=0):
    path = r'TSRxCPxPSI.csv'
    df = pd.read_csv(path)
    angles = np.arange(-5,6,1)
    TSR = df.iloc[:,0]
    diff1 = np.abs(TSR - LAMBDA)
    diff2 = np.abs(angles - PHI)
    id1 = np.argmin(diff1)
    id2 = np.argmin(diff2)+1

    return df.iloc[id1,id2]

class Aerodynamics:
    def __init__(self, **params):

        self.Area =params['Area']
        self.R =params['R']
        self.RHO =params['RHO']
        
    def TAU_r(self, OMEGA_r, v, PHI_m):
        
        Area = self.Area
        R = self.R
        RHO = self.RHO
        
        LAMBDA = OMEGA_r*R/v
        C_p = Cp_calc(LAMBDA, PHI_m)
        TAU = RHO*Area*C_p*(v**3)/(2*OMEGA_r)
        
        return TAU