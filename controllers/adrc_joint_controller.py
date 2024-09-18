import numpy as np
from observers.eso import ESO
from .controller import Controller
from .pd_controller import PDDecentralizedController
from models.manipulator_model import ManiuplatorModel

class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp, i):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.i = i

        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.reshape([0, b, 0], (-1 , 1)) # creates a vertical vector
        L = np.reshape([3*p, 3*p**2, p**3], (-1, 1))
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

        self.controller_pd = PDDecentralizedController(self.kp, self.kd)    
        self.model = ManiuplatorModel(Tp, 1.00, 0.05)

    def set_b(self, b):
        self.b = b
        self.eso.set_B(np.reshape([0, b, 0], (-1, 1)))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        # calculate b
        q, q_dot = x
        m_ii = np.linalg.inv(self.model.M([q, q_dot, 0, 0]))[self.i, self.i]
        self.set_b(m_ii)

        # Get extended state estimate from ESO
        q_eso, q_dot_eso, f_eso = self.eso.get_state() 

        # calculate PD control signal with ADRC
        v = self.controller_pd.calculate_control(q_eso, q_dot_eso, q_d, q_d_dot, q_d_ddot)
        u = (v - f_eso) / self.b

        # update ESO
        q, qdot = x
        self.eso.update(q, u)

        return u