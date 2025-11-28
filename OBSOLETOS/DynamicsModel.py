import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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

class HawtDynamics:
    def __init__(self, X=None, **params):

        
        self.X=np.array([0.1,0.1,0])
        self.Area =params['Area']
        self.B_dt =params['B_dt']
        self.B_r =params['B_r']
        self.B_g =params['B_g']
        self.ETA_gc =params['ETA_gc']
        self.ETA_dt =params['ETA_dt']
        self.f_s =params['f_s']
        self.J_r =params['J_r']
        self.J_g =params['J_g']
        self.K_dt =params['K_dt']
        self.k_i =params['k_i']
        self.k_p =params['k_p']
        self.LAMBDA_opt =params['LAMBDA_opt']
        self.N_g =params['N_g']
        self.OMEGA_nom =params['OMEGA_nom']
        self.OMEGA_var =params['OMEGA_var']
        self.PI =params['PI']
        self.P_r =params['P_r']
        self.R =params['R']
        self.RHO =params['RHO']
        self.T_s = params['T_s']
        self.ZETA = params['ZETA']
        
        self.ctrl_mode = 'mode1'
        self.mode = np.array([])
        self.OMEGA_r = np.array([])
        self.OMEGA_g = np.array([])
        self.TAU_g = np.array([0])
        self.TAU_r = np.array([0])  
        self.U = None
        self.X_pitch = np.array([0,0]) 
        self.dX_ptich = np.array([0,0])
        self.PHI_m = np.array([0])
        self.PHI_ref = np.array([0])
        self.e = np.array([0])


    def TAUr_calc(self, OMEGA_r, PHI_m, v):

        Area = self.Area
        R = self.R
        RHO = self.RHO
        PI = self.PI

        self.PHI_m = np.append(self.PHI_m, PHI_m)

        LAMBDA = OMEGA_r*R/v
        C_p = Cp_calc(LAMBDA, PHI_m)
        TAU_r = RHO*PI*(R**3)*C_p*(v**2)/2
        self.TAU_r = np.append(self.TAU_g, TAU_r)
        return TAU_r
    
    def TAUg_calc(self, OMEGA_g, u_k=0):
        
        Area = self.Area
        ETA_gc = self.ETA_gc
        e_bfr = self.e[-1]
        k_p = self.k_p
        k_i = self.k_i
        N_g = self.N_g
        OMEGA_nom = self.OMEGA_nom
        OMEGA_var = self.OMEGA_var
        PHI_m = self.PHI_m[-1]
        P_r = self.P_r
        PHI_ref = self.PHI_ref[-1]
        R = self.R
        RHO = self.RHO
        f_s = self.f_s

        LAMBDA = self.LAMBDA_opt + u_k
        
        C_p = Cp_calc(LAMBDA,PHI_m)
        K_mppt = RHO*Area*(R**3)*C_p/(2*(LAMBDA**3))
        TAU_g = K_mppt*((OMEGA_g/N_g)**2)
        P_g = ETA_gc*OMEGA_g*TAU_g

        if P_g >= P_r or OMEGA_g >= OMEGA_nom:
            self.ctrl_mode = 'mode2'
        if OMEGA_g < OMEGA_nom - OMEGA_var:
            self.ctrl_mode = 'mode1'
            #self.PHI_ref = np.append(self.PHI_ref,0)

        if self.ctrl_mode == 'mode1':
            self.mode = np.append(self.mode,1)
            self.K_mppt = K_mppt
            self.TAU_g = np.append(self.TAU_g, TAU_g)

        if self.ctrl_mode == 'mode2':
            self.mode = np.append(self.mode,2)
            TAU_g  = P_r/(ETA_gc*OMEGA_g)
            e = OMEGA_g - OMEGA_nom
            PHI_ref = self.PHI_ref[-1] + k_p*e + (k_i*f_s - k_p)*e_bfr
            self.e = np.append(self.e, e)
            self.PHI_ref = np.append(self.PHI_ref, PHI_ref)
            self.TAU_g = np.append(self.TAU_g, TAU_g)

        return self.TAU_g[-1], self.PHI_ref[-1]

    def Model_Input(self, PHI_m, v, u_k=0):
        
        OMEGA_r, OMEGA_g = self.X[0], self.X[1]
        TAU_r = self.TAUr_calc(OMEGA_r, PHI_m, v)
        TAU_g, PHI_ref = self.TAUg_calc(OMEGA_g, u_k)
        self.PHI_ref = np.append(self.PHI_ref, PHI_ref)
        
        du = np.array([TAU_r,TAU_g])
        

        self.U = du
        
        return du
        
    def Model_Dynamics(self, t, X, U):

        B_dt, B_r, B_g = self.B_dt, self.B_r, self.B_g
        J_r, J_g, N_g = self.J_r, self.J_g, self.N_g
        ETA_dt, K_dt = self.ETA_dt, self.K_dt 

        A=np.array([[        -(B_dt+B_r)/J_r,                          B_dt/(N_g*J_r),                -K_dt/J_r],
                    [(B_dt*ETA_dt)/(N_g*J_g), -(((B_dt*ETA_dt)/(N_g**2))+B_g)*(1/J_g),  (K_dt*ETA_dt)/(N_g*J_g)],
                    [                      1,                                  -1/N_g,                        0]])
        
        B = np.array([[1/J_r,      0],
                      [    0, -1/J_g],
                      [    0,      0]])
        
        dX = (A@X.reshape(-1,1))+(B@U.reshape(-1,1))
        self.dX = dX.flatten()
        return dX.flatten()
        
    def Model_iterate(self, u, dt):
        #DOP853
        sol = solve_ivp(self.Model_Dynamics, [0, dt], self.X, 
            args=(u.flatten(),), method="RK23", t_eval=[dt])
        self.X = sol.y[:, -1]
        self.OMEGA_r = np.append(self.OMEGA_r, self.X[0])
        self.OMEGA_g = np.append(self.OMEGA_g, self.X[1])
        return self.X
    
    def Model_Calc(self, v, PHI_m=0, u_k=0):
        U = self.Model_Input(PHI_m, v, u_k)
        X = self.Model_iterate(U,1)
        return X