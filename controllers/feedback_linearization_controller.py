import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp, 1.0, 0.05)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        q = x[:2]
        q_dot = x[2:]

        # PD Feedback
        e = q_r - q
        e_dot = q_r_dot - q_dot
    
        Kp = np.array([[5], [25]]) * np.eye(2) 
        Kd = 1 * np.eye(2)
        v = q_r_ddot + Kd @ e_dot + Kp @ e

        # No Feedback
        #v = q_r_ddot

        tau = self.model.M(x) @ v + self.model.C(x) @ q_r_dot
        return tau
