import numpy as np

from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp, 1.0, 0.05)
        self.Kp = Kp
        self.Kd = Kd
        
        p1, p2 = p
        self.L = np.array([[3*p1, 0], [0, 3*p2], [3*p1**2, 0],
                           [0, 3*p2**2], [p1**3, 0], [0, p2**3]])
        W = np.block([[np.eye(2), np.zeros((2, 4))]])
        A = self.get_A(q0[:2], q0[2:])
        B = self.get_B(q0[:2], q0[2:])
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        
        self.update_params(q0[:2], q0[2:])

    def get_A(self, q, q_dot):
        x = np.concatenate([q, q_dot])
        M = self.model.M(x)
        C = self.model.C(x)
        A = np.block([[np.zeros((2, 2)), np.eye(2), np.zeros((2, 2))], 
                      [np.zeros((2, 2)), -np.linalg.inv(M) @ C, np.eye(2)],
                      [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]
                    ])
        return A
    
    def get_B(self, q, q_dot):
        x = np.concatenate([q, q_dot])
        M = self.model.M(x)
        B = np.block([[np.zeros((2, 2))], 
                      [np.linalg.inv(M)],
                      [np.zeros((2, 2))]
                    ])
        return B
    
    

    def update_params(self, q, q_dot):
        self.eso.A = self.get_A(q, q_dot)
        self.eso.B = self.get_B(q, q_dot)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        self.update_params(x[:2], x[2:])
        z = self.eso.get_state()

        q_eso = z[:2]
        q_dot_eso = z[2:4]
        f_eso = z[4:]

        e = q_d - q_eso
        e_dot = q_d_dot - q_dot_eso

        v = q_d_ddot + self.Kp @ e + self.Kd @ e_dot
        u = self.model.M(x) @ (v - f_eso) + self.model.C(x) @ q_dot_eso
        
        self.eso.update(np.reshape(q_eso, (-1, 1)), np.reshape(u, (-1, 1)))
        return u