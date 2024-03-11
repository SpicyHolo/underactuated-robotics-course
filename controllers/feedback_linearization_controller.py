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
        v = np.reshape(q_r_ddot, (2,1))
        q_dot = np.array([[q1_dot], [q2_dot]])
        M = self.model.M(x)
        C = self.model.C(x)

        tau = M @ v + C @ q_dot

        return tau
