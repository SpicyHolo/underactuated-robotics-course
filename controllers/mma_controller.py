import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        model1 = self.model = ManiuplatorModel(Tp, 0.1, 0.05)
        model2 = self.model = ManiuplatorModel(Tp, 0.01, 0.01)
        model3 = self.model = ManiuplatorModel(Tp, 1., 0.3)

        self.models = [model1, model2, model3]
        self.i = 0

        self.u_prev = np.zeros(2)
        self.x_prev = np.zeros(4)

    def choose_model(self, x):
        # Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        e = [x - self.predict_model(model, self.x_prev, self.u_prev) for model in self.models]
        e_norm = np.linalg.norm(e, axis=(1, 2))
        print(e_norm)
        min_error_i = np.argmin(e_norm)
        self.i = min_error_i


    def predict_model(self, model, x, u):
        q_dot = np.reshape(x[2:], (2, 1))
        u = np.reshape(u, (2, 1))
        q_ddot = np.linalg.inv(model.M(x)) @ (u - model.C(x) @ q_dot)
        x_dot = np.concatenate((q_dot, q_ddot))
        return x + x_dot * model.Tp
        

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        # PD Feedback controller
        e = q_r - q
        e_dot = q_r_dot - q_dot
        Kp = np.array([[5], [25]]) * np.eye(2) 
        Kd = np.eye(2) * 1
        v = q_r_ddot + Kd @ e_dot + Kp @ e

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        
        self.u_prev = u
        self.x_prev = x

        return u
