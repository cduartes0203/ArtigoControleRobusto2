import numpy as np

class DriveTrain:
    def __init__(self, **params):

        self.RHO = params['rho'] 
        self.A = params['Ar'] 
        self.R = params['Rr'] 
        self.Bdt = params['Bdt']
        self.Kdt = params['Kdt']
        self.Jr = params['Jr']
        self.Jg = params['Jg']
        self.lmbd_opt = params['Lambda_opt']
        self.Cp_star = params['Cp_Max']
        self.U = np.array([0,0])
        self.PD = np.array([0])
        self.PG = np.array([0])
        self.ED = np.array([0])
        self.EG = np.array([0])

        self.Tg_r = None
        self.B_r = None
        self.tg = None
        self.xi = None
        self.Wn = None
        self.dTg = None
        self.dB = None
        self.ddB = None
        self.dO = None
        self.dWg = None
        self.dWr = None

        self.Tg = None
        self.B = None
        self.O = None
        self.Wg = None
        self.Wr = None

        self.X = np.array([self.Tg,self.B,self.dB,self.O,self.Wg,self.Wr])
        self.dX = np.array([self.dTg,self.dB,self.ddB,self.dO,self.dWg,self.dWr])


    def set_state(self, X):
        """
        Define the current state of the plant. 
        
        x : array-like
            Current state [omega_r, omega_g, theta]
        """
        self.X = X.flatten()

    def dynamics(self, t, X, U):


        Bdt = self.Bdt 
        Kdt = self.Kdt 
        Jr = self.Jr 
        Jg = self.Jg 
        tg = self.tg
        xi = self.xi
        Wn = self.Wn
        one = np.array([[1]])

        A11 = (-1/tg)*one
        A54 = -(Wn**2)*one 
        A55 = -2*xi*Wn
        
        A0=np.array([[A11,    0,   0,           0,           0,           0],
                     [  0,    0,  one,           0,           0,           0],
                     [  0,  A54, A55,           0,           0,           0],
                     [  0,    0,   0,           0,        1/Ng,           1],
                     [a71,    0,   0, Kdt/(Jg*Ng),         a77, Bdt/(Ng*Jg)],
                     [   0, a84,   0,   -Kdt/(Jr), Bdt/(Ng*Jg),         a88]])
        
        B = np.array([[1/Jr,      0],
                      [    0, -1/Jg],
                      [    0,     0]])
        
        dX = (A@X.reshape(-1,1))+(B@U.reshape(-1,1))
        self.dX = dX.flatten()
        return dX.flatten()
    
    def compute_input(self, v, X, uk=0):
        """
        Compute control input u as u = Kx + uf, where
        uf is a feedforward term.
        
        Parameters:
        x: numpy array
            Current state [x, y, theta, vx, vy, w]
        K: numpu array matrix-alike
            Gain matrix.
        """
        
        wr = X[0]
        wg = X[1]
        rho = self.rho
        A = self.A
        R = self.R
        lmbd_star = self.lmbd_opt + uk
        lmbd = wr*R/v
        Cp_star = self.Cp_star
        if lmbd_star != self.lmbd_opt:
            Cp_star = Cp_calc(lmbd_star)
        Cp = Cp_calc(lmbd)
        self.cp_.append(Cp_star)
        self.lmbd_.append(lmbd_star)
        K_mppt = 0.5 * rho * A * (R**3) * Cp_star/(lmbd_star**3)
        #print(lmbd_star,Cp_star,K_mppt)
        tR = 0.5*rho*A*Cp*(v**3)/wr
        tG = K_mppt*(wg**2)
        du = np.array([tR,tG])
        self.U = du
        self.K_mppt = np.append(self.K_mppt,K_mppt)
        
        return du
    
    def iterate(self, u, dt):
        """
        Propagate the system forward in time by 'dt' using 
        numerical integration.
        
        Parameters:
        u : array-like
            Input vector [u1, u2]
        dt : float
            Time step
        """
        #DOP853
        sol = solve_ivp(self.dynamics, [0, dt], self.X, 
            args=(u.flatten(),), method="DOP853", t_eval=[dt])
        self.X = sol.y[:, -1]
        return self.X
    
    def compute_output(self):
        X = self.X.reshape(1,-1)
        dX = self.dX.reshape(1,-1)
        B = self.Bdt
        tau_r, tau_g = self.U
        w_r, w_g, _ = self.X
        PG = tau_g*w_g
        PD = B*((w_r-w_g)**2)

        #if self.k>=300:
        EG = self.EG[-1]+PG
        ED = self.ED[-1]+PD

        self.PD = np.append(self.PD,PD)
        self.PG = np.append(self.PG,PG) 
        self.ED = np.append(self.ED,ED)
        self.EG = np.append(self.EG,EG) 
        self.X_ = np.append(self.X_,X,axis=0)  
        self.dX_ = np.append(self.dX_,dX,axis=0)  
        if self.k == 1:
            self.PD = np.delete(self.PD,0)
            self.PG = np.delete(self.PG,0) 
            self.ED = np.delete(self.ED,0)
            self.EG = np.delete(self.EG,0) 

        self.k = self.k + 1
