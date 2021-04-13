import numpy as np

from scipy.optimize import minimize

class HuberScore:
    """Robust Huber score function."""
    def __init__(self, delta=1.5):
        self._delta = delta

    def evaluate(self, z):
        if abs(z) >= self._delta:
            return self._delta * abs(z) - pow(self._delta, 2) / 2.0
        else:
            return pow(z, 2) / 2.0

class RobustKalmanFilter(object):
    def __init__(self, F, B, H, x0, P0, Q0, R0, use_robust_estimation=False, robust_score=HuberScore(delta=1.5)):
        """
        Initialize robust kalman filter.
        
        Args:
            F: State transition matrix
        """
        self.F = F.copy()
        self.B = B.copy() if B is not None else None
        self.H = H.copy()
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q0.copy()
        self.R = R0.copy()
        
        self.use_robust_estimation = use_robust_estimation
        
        self.robust_score = robust_score
        
    def time_update(self, inputs=None):
        """
        Time propagation of the system model
        
        Args:
            inputs: Model inputs if any.
        """
        if inputs is None:
            self.x = np.matmul(self.F, self.x)
        else:
            self.x = np.matmul(self.F, self.x) + np.matmul(self.B, inputs)
        
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
    
    def measurement_update(self, measurements):
        """
        Measurement update. Not that time update must preceded the measurement update
        for valid estimation results
        
        Args:
            measurements: Observations of measured quantities.
        """
        # Residual or inovation
        self.inovation = measurements - np.matmul(self.H, self.x)
        
        Pinov = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R
        # Compute the Kalman Gain
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(Pinov))

        if self.use_robust_estimation:
            epsilon_covariance = np.bmat([[self.P, np.zeros((self.P.shape[0], self.R.shape[1]))],
                                          [np.zeros((self.R.shape[0], self.P.shape[1])), self.R]])
    
            S = np.linalg
            Sinv = np.linalg.inv(S)
            
            Y = np.matmul(Sinv, np.vstack((self.x, measurements)))
            X = np.matmul(Sinv, np.vstack((np.eye(self.x.shape[0]), self.H)))
            
            res = minimize(lambda xx: self._m_estimate_criterion(xx, Y, X), self.x, method='nelder-mead')

            self.x = res.x[np.newaxis].T
        
        else:
            self.x = self.x + np.matmul(K, self.inovation)
        
        self.P = self.P - np.matmul(np.matmul(K, self.H), self.P)
    
    def _m_estimate_criterion(self, x, Y, X):
        """Criterion for robust state estimation"""
        crit = 0.0
        for i in range(Y.shape[0]):
            crit += self.robust_score.evaluate(Y[i, :] - np.matmul(X[i, :], x))
            #crit += (Y[i, :] - np.matmul(X[i, :], x))**2

        return crit
        