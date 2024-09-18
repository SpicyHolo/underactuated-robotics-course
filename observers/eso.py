from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def mat_or_scal_multiply(self, B, u):
        if np.isscalar(u):
            return B * u
        else:
            return B @ u


    def update(self, q, u):
        self.states.append(copy(self.state))
        
        z = np.reshape(self.state, (-1, 1))
        state_dot = self.A @ z + self.mat_or_scal_multiply(self.B, u) + self.L @ (q - self.W @ z)


        self.state = self.state + self.Tp * state_dot.ravel()


    def get_state(self):
        return self.state
