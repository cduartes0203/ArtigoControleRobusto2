import numpy as np
import pandas as pd

def Cp_calc(LAMBDA=0, PHI=0):
    path = r'Datasets/Cp_X_lambda.csv'
    df = pd.read_csv(path)
    dist = np.abs(df['lambda'].values - LAMBDA)

    return (df['cp'].values[np.argmin(dist)])

class TorqueController:
    def __init__(self, **params):

        self.Area =params['Area']
        self.R =params['R']
        self.RHO =params['RHO']
        self.PHI_opt = params['PHI_opt']
        self.LAMBDA_opt =params['LAMBDA_opt']
        self.K_mppt = None
        self.TAU = None
        self.P_g = None

    def TAU_g(self, OMEGA_g, u_k=0, PHI=None):
        
        Area = self.Area
        R = self.R
        RHO = self.RHO
        LAMBDA = self.LAMBDA_opt + u_k

        if PHI == None:
            PHI = self.PHI_opt
        
        C_p = Cp_calc(LAMBDA)
        K_mppt = RHO*Area*(R**3)*C_p/(2*(LAMBDA**3))
        TAU = K_mppt*(OMEGA_g**2)
        P_g = TAU*OMEGA_g
        self.P_g = P_g
        self.K_mppt = K_mppt
        self.TAU = TAU
        return TAU