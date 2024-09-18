import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.adrc_controller import ADRController

from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.01
end = 5

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

b_est_1 = 2.5
b_est_2 = 0.3

kp_est_1 = 15
kp_est_2 = 20
kd_est_1 = 20
kd_est_2 = 30
p1 = 20. 
p2 = 30. 

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])
controller = ADRController(Tp, params=[[b_est_1, kp_est_1, kd_est_1, p1, q1_0],
                                       [b_est_2, kp_est_2, kd_est_2, p2, q2_0]])

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso1 = np.array(controller.joint_controllers[0].eso.states)
eso2 = np.array(controller.joint_controllers[1].eso.states)

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].plot(T, eso1[:, 0], label="q est.")
axes[0].plot(T, Q[:, 0], 'r', label="q")
axes[0].set_title("Joint 0, position ")
axes[1].plot(T, eso1[:, 1], label="q_dot est.")
axes[1].plot(T, Q[:, 2], 'r', label="q_dot")
axes[1].set_title("Joint 0, speed")
axes[2].plot(T, eso2[:, 0], label="q_est.")
axes[2].plot(T, Q[:, 1], 'r', label="q")
axes[2].set_title("Joint 1, position ")
axes[3].plot(T, eso2[:, 1], label="q_dot est.")
axes[3].plot(T, Q[:, 3], 'r', label="q_dot")
axes[3].set_title("Joint 1, speed")
for ax in axes:
    ax.legend()
plt.tight_layout()

fix, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].plot(T, Q[:, 0], 'r', label="q1")
axes[0].plot(T, Q_d[:, 0], 'b', label="q1 desired")
axes[1].plot(T, Q[:, 1], 'r', label="q2")
axes[1].plot(T, Q_d[:, 1], 'b', label="q2 desired")
axes[2].plot(T, u[:, 0], 'r', label="u_1")
axes[2].plot(T, u[:, 1], 'b', label="u_2")
for ax in axes:
    ax.legend()
plt.show()
