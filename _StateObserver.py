import numpy as np
import cvxpy as cp # type: ignore
from scipy.linalg import sqrtm
import mosek

class StateObserver:
    def __init__(self, Ts=1, c=1, D=1, B=1e-3, **params):
        """
        Inicializa o Observador de Estado.
        ts: Período de amostragem (Ts ou dt).
        """
        self.Ts = Ts
        self.c = c
        self.X = np.array([D,B]).reshape(-1,1)
        self.L = None
        self.H = np.identity(2) + Ts*np.array([[0,1],[0,-c]])
        self.C = np.array([[1, 0]])
    

    def solve_lmi_gain(self, Q=np.diag([1, 2.5e-5]), R=np.array([[100.0]])):
        """
        Calcula o ganho ótimo do observador L usando a solução LMI para o problema LQ
        proposto na Seção 4.4.3 do artigo.

        Args:
            Q (np.array): Matriz de custo do estado (process noise covariance).
            R (np.array): Matriz de custo da medição (measurement noise covariance).
        """
        H = self.H
        C = self.C
        n = H.shape[0] # Dimensão do estado (n=2)
        p = C.shape[0] # Dimensão da saída (p=1)

        # ---- Formulação do Problema LMI (Seção 4.4.3) ----
        # 1. Definir as variáveis da LMI
        # P: Matriz de Lyapunov (simétrica, definida positiva)
        P = cp.Variable((n, n), PSD=True)
        # Y: Variável de mudança para L (L = inv(P) * Y^T)
        Y = cp.Variable((p, n))
        # W: Limite superior para a função de custo
        W = cp.Variable((n + p, n + p), symmetric=True)

        # 2. Definir as matrizes de custo estendidas M e N (Equação 56)
        M = np.vstack([sqrtm(Q), np.zeros((p, n))])
        N = np.vstack([np.zeros((n, p)), sqrtm(R)])

        # 3. Definir as restrições (constraints) da LMI
        #constraints = [P >> 1e-16*np.eye(n)] # P deve ser definida positiva
        constraints = [P >> 1e-8]

        # Restrição de estabilidade (Equação 58, adaptada da condição de Lyapunov)
        # Esta é a LMI principal que garante a convergência do erro de estimação.
        lmi_stability = cp.bmat([
            [           (-P+Q), (H.T @ P - C.T @ Y)],
            [(P.T @ H - Y.T @ C),                -P]
        ])
        constraints += [lmi_stability << -1e-8]

        # Restrição da função de custo (Equação 61)
        lmi_cost = cp.bmat([
            [                      W, (M @ P + N @ Y)],
            [(P.T @ M.T + Y.T @ N.T),               P]
        ])
        constraints += [lmi_cost >> 1e-8]
        
        # 4. Definir o problema de otimização (Equação 62)
        # Minimizar o traço de W, que minimiza a função de custo do erro.
        objective = cp.Minimize(cp.trace(W))
        problem = cp.Problem(objective, constraints)

        # 5. Resolver o problema
        problem.solve(solver=cp.MOSEK)
        
        if problem.status not in ["infeasible", "unbounded"]:
            # Calcular o ganho L a partir das variáveis da solução
            # L = (inv(P) * Y^T)
            P_val = P.value
            Y_val = Y.value
            L_val = np.linalg.inv(P_val) @ Y_val.T
            self.L = (L_val.flatten()).reshape(-1,1)
            #print(f"Ganho L calculado com sucesso: {self.L}")
            
        else:
            print(f"Falha ao resolver a LMI. Status: {problem.status}")
            

    def update(self, y_k):
        """
        Atualiza o estado estimado usando a equação do observador (Equação 70).

        Args:
            y_k (float): A medição atual da degradação (energia dissipada).
        
        Returns:
            np.array: O novo vetor de estado estimado x_hat_{k+1}.
        """
        if self.L is None:
            raise RuntimeError("O ganho L não foi calculado. Chame 'solve_lmi_gain' primeiro.")

        
        # Próximo estado estimado
        self.X = self.H @ self.X + self.L @ (y_k - (self.C @ self.X))
        
        return self.X[0], self.X[1]
    
