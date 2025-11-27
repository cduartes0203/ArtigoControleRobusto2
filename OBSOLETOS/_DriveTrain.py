import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def Cp_calc(lmbd_in=0):
    path = r'Datasets/Cp_X_lambda.csv'
    df = pd.read_csv(path)
    dist = np.abs(df['lambda'].values - lmbd_in)

    return (df['cp'].values[np.argmin(dist)])

class DriveTrain:
    def __init__(self, **params):

        self.rho = params['rho'] 
        self.A = params['Ar'] 
        self.R = params['Rr'] 
        self.Bdt = params['Bdt']
        self.Kdt = params['Kdt']
        self.Jr = params['Jr']
        self.Jg = params['Jg']
        self.lmbd_opt = params['Lambda_opt']
        self.Cp_star = params['Cp_Max']
        self.X = np.array([0,0,0])
        self.dX = np.array([0,0,0])
        self.U = np.array([0,0])
        self.PD = np.array([0])
        self.PG = np.array([0])
        self.ED = np.array([0])
        self.EG = np.array([0])
        self.K_mppt = np.array([[0]])
        self.X_ = np.array([[0,0,0]])
        self.dX_ = np.array([[0,0,0]])
        self.k = 1
        self.tR = np.array([])
        self.tG = np.array([])
        self.cp_=[]
        self.lmbd_=[]

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
        
        A=np.array([[-Bdt/Jr,  Bdt/Jr, -Kdt/Jr],
                    [ Bdt/Jg, -Bdt/Jg,  Kdt/Jg],
                    [    1,    -1,     0]])
        
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
        self.tR = np.append(self.tR,tR)
        self.tG = np.append(self.tG,tG)
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

    def all_plots(self):
        PG, PD, EG, ED, K_mppt, wr, wg, ot, dwr, dwg, dot, tR, tG= [
            {
                'y_arrays': [self.PG],
                'x_arrays': None,
                'title': 'Generated Power',
                'xname': None,
                'yname': r'$P_{G}$',
                'legend_labels': None
            },
            {
                'y_arrays': [self.PD],
                'x_arrays': None,
                'title': 'Dissipated Power',
                'xname': None,
                'yname': r'$P_{D}$',
                'legend_labels': None
            },
            {
                'y_arrays': [self.EG],
                'x_arrays': None,
                'title': 'Cumulative Generated Energy',
                'xname': None,
                'yname': r'$E_{G}$',
                'legend_labels': None
            },
            {
                'y_arrays': [self.ED],
                'x_arrays': None,
                'title': 'Cumulative Dissipated Energy',
                'xname': None,
                'yname': r'$E_{D}$',
                'legend_labels': None
            },
            {
                'y_arrays': [self.K_mppt[1:]],
                'x_arrays': None,
                'title': r'$K_{mppt}$ gain consecutively',
                'xname': None,
                'yname': r'$K_{mppt}$ gain',
                'legend_labels': None
            },
            {
                'y_arrays': [self.X_[:,0]],
                'x_arrays': None,
                'title': r'$\omega_{r}$ consecutively',
                'xname': None,
                'yname': r'$\omega_{r}$ (rad/s)',
                'legend_labels': None
            },
            {
                'y_arrays': [self.X_[:,1]],
                'x_arrays': None,
                'title': r'$\omega_{g}$ consecutively',
                'xname': None,
                'yname': r'$\omega_{g}$ (rad/s)',
                'legend_labels': None
            },
            {
                'y_arrays': [self.X_[:,2]],
                'x_arrays': None,
                'title': r'$\theta_{ts}$ consecutively',
                'xname': None,
                'yname': r'$\theta_{ts}$ (rad)',
                'legend_labels': None
            },
            {
                'y_arrays': [self.dX_[:,0]],
                'x_arrays': None,
                'title': r'$\dot{\omega}_{r}$ consecutively',
                'xname': None,
                'yname': r'$\dot{\omega}_{r}$ (rad/s)',
                'legend_labels': None
            },
            {
                'y_arrays': [self.dX_[:,1]],
                'x_arrays': None,
                'title': r'$\dot{\omega}_{g}$ consecutively',
                'xname': None,
                'yname': r'$\dot{\omega}_{g}$ (rad/s)',
                'legend_labels': None
            },
            {
                'y_arrays': [self.dX_[:,2]],
                'x_arrays': None,
                'title': r'$\dot{\theta}_{ts}$ consecutively',
                'xname': None,
                'yname': r'$\dot{\theta}_{ts}$ (rad)',
                'legend_labels': None
            },
        {
            'y_arrays': [self.tR],
            'x_arrays': None,
            'title': r'$\dot{\tau}_{r}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\tau}_{r}$ (rad)',
            'legend_labels': None
        },
        {
            'y_arrays': [self.tG],
            'x_arrays': None,
            'title': r'$\dot{\tau}_{g}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\tau}_{g}$ (rad)',
            'legend_labels': None
        },
            ]
        
        return EG, PG, ED, PD, K_mppt, wr, wg, ot, dwr, dwg, dot, tR, tG

    