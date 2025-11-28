import numpy as np
from scipy.integrate import solve_ivp

class PitchController:
    def __init__(self, PHI_ref=np.array([0.1]), **params):

        self.PHI_ref = PHI_ref
        self.ZETA = params['ZETA']
        self.OMEGA_n = params['OMEGA_n']

        self.X_pitch = np.array([0,0]) 
        self.dX_ptich = np.array([0,0]) 

    def Pitch_input(self):

        dU_pitch = self.PHI_ref
        
        return dU_pitch
        
    def Pitch_dynamics(self, t, X, U):
            
        #U = self.PHI_ref
        #X = self.X_ptich
        Z = self.ZETA
        Wn = self.OMEGA_n

        A = np.array([[-2*Wn*Z, Wn**2],
                    [      1,     0]])
        B = np.array([[Wn**2],
                    [    0]])
        
        #print(U, '\n', U.reshape(-1,1))
        dX = A@X.reshape(-1,1) + B@U.reshape(-1,1)
        self.dX_ptich = dX.flatten()
        return dX.flatten()
        
    def Pitch_iterate(self, u, dt):
        #DOP853
        sol = solve_ivp(self.Pitch_dynamics, [0, dt], self.X_pitch, 
            args=(u.flatten(),), method="RK23", t_eval=[dt])
        self.X_pitch = sol.y[:, -1]
        return self.X_pitch
    
    def Pitch_Calc(self):
        U_pitch = self.Pitch_input()
        X_pitch = self.Pitch_iterate(U_pitch,1)
        return X_pitch[1] 
    