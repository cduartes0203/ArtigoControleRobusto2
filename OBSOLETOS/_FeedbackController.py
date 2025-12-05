import numpy as np
import pandas as pd
import cvxpy as cp # type: ignore
import mosek
from scipy.linalg import sqrtm


class FeedbackController:
    def __init__(self, **params):
        """
        Inicializa o Controlador de Feedback (Degradation-Rate Controller).

        Args:
            gamma_min (float): O valor mínimo esperado para o parâmetro de incerteza ~γ.
            gamma_max (float): O valor máximo esperado para o parâmetro de incerteza ~γ.
        """
        # Ganhos do controlador K = [K1, K2]
        self.K = None
        self.K1 = None
        self.K2 = None

        #lambda ótimo
        self.lmbd_opt = params['Lambda_opt']
        self.vwL = params['Vmin']
        self.vwU = params['Vmax']
        self.Bdt = params['Bdt']
        self.Rr = params['Rr']

        # Parâmetros para o LQR Robusto
        self.gamma_min = (self.Bdt*(self.vwL**2))*2*self.lmbd_opt/(self.Rr**2)
        self.gamma_max = (self.Bdt*(self.vwU**2))*2*self.lmbd_opt/(self.Rr**2)

        # Estados internos do controlador
        self.u_k = 1 # Valor inicial para u_{k-1}
        self.z_k = 1e-3         # Valor inicial para o erro integral z_k
        self.lmbd_star = self.lmbd_opt+self.u_k

    
        
        
    def solve_lqr_gain(self, Q=np.diag([1, 1]), R=np.array([[1e-5]])):
        """
        Calcula os ganhos ótimos K1 e K2 usando a solução LMI para o problema
        de LQR Robusto descrito na Seção 4.4.2 do artigo.
        
        Args:
            Q (np.array): Matriz de custo do estado (2x2).
            R (np.array): Matriz de custo do controlo (1x1).
        """
        n = 2 # Dimensão do estado x_k = [u_{k-1}, z_k]
        m = 1 # Dimensão da entrada u_k

        # --- Formulação do Problema LMI (Seção 4.4.2) ----
        # 1. Definir as variáveis da LMI
        P = cp.Variable((n, n), PSD=True)
        Y = cp.Variable((m, n))
        W = cp.Variable((n + m, n + m), symmetric=True)

        # 2. Definir as matrizes de custo estendidas M e N (Equação 39)
        M = np.vstack([sqrtm(Q), np.zeros((m, n))])
        N = np.vstack([np.zeros((n, m)), sqrtm(R)])

        # 3. O sistema politópico tem 2 vértices (extremos de ~γ)
        # Vértice 1: Corresponde a gamma_min
        # Vértice 2: Corresponde a gamma_max
        A_vertices = [
            np.array([[0, 0], [self.gamma_min, 1]]),
            np.array([[0, 0], [self.gamma_max, 1]])
        ]
        B = np.array([[1], [0]])

        # 4. Definir as restrições da LMI para TODOS os vértices
        constraints = [P >> 1e-8] # P deve ser definida positiva

        for A in A_vertices:
            # Restrição de estabilidade (Equação 44)
            lmi_stability = cp.bmat([
                [               (-P+Q), (A @ P - B @ Y)],
                [(P @ A.T - Y.T @ B.T),              -P]
            ])
            constraints += [lmi_stability << -1e-8]

        # Restrição da função de custo (Equação 47)
        lmi_cost = cp.bmat([
            [                    W, (M @ P + N @ Y)],
            [(P @ M.T + Y.T @ N.T),               P]
        ])
        constraints += [lmi_cost >> 1e-8]
        
        # 5. Definir e resolver o problema de otimização
        objective = cp.Minimize(cp.trace(W))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK,verbose=False)

        # 6. Calcular os ganhos K a partir da solução
        if problem.status in ["infeasible", "unbounded"] or P.value is None:
            print(f"Falha ao resolver a LMI do LQR. Status: {problem.status}")
            return None

        P_val = P.value
        Y_val = Y.value
        
        # K = Y * inv(P)
        K_val = Y_val @ np.linalg.inv(P_val)
        self.K = K_val.flatten()
        self.K1, self.K2 = self.K[0], self.K[1]
        
        #print(f"Ganhos LQR calculados com sucesso: K1 = {self.K1:.4f}, K2 = {self.K2:.4f}")
        
        
    def update(self, beta_hat, beta_ref, dt=1):
        """
        Calcula a ação de controlo u_k (ou seja, Δλ) com base na lei de controlo
        da Equação (83).

        Args:
            beta_hat (float): A taxa de degradação atual estimada (saída do observador).
            beta_ref (float): A taxa de degradação de referência (saída do conversor).
            dt (float): O período de amostragem.
        
        Returns:
            float: A ação de controlo calculada, u_k.
        """
        if self.K is None:
            raise RuntimeError("Os ganhos K não foram calculados. Chame 'solve_lqr_gain' primeiro.")
            
        # 1. Atualizar o erro integral (Equação 84, em tempo discreto)
        self.z_k = self.z_k+ (beta_hat - beta_ref) * dt
        
        # 2. Calcular a ação de controlo u_k (Equação 83)
        u_k = -self.K1 * self.u_k - self.K2 * self.z_k
        
        # 3. Atualizar o estado da ação anterior para a próxima iteração
        self.lmbd_star=self.lmbd_opt+u_k
        self.u_k = u_k[0]

        return self.lmbd_star, self.u_k
       