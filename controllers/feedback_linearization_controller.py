import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        
        # FEEDBACK PD
        e = q_r - [q1, q2]
        e_dot = q_r_dot - [q1_dot, q2_dot]
        
        Kd = np.array([[20, 0], [0, 20]])
        Kp = np.array([[20, 0], [0, 20]])
        v = np.reshape(q_r_ddot, (2, 1))+ Kd @ np.reshape(e_dot, (2, 1)) + Kp @ np.reshape(e, (2, 1))

        # NO FEEDBACK
        #v = q_r_ddot

        M = self.model.M(x)
        C = self.model.C(x)
        tau = M @ np.reshape(v, (2, 1)) + C @ np.reshape(q_r_dot, (2, 1))
        print(tau)
        return tau
