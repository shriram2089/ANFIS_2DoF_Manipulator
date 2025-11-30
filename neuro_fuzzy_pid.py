# simulate_nfpid.py
import numpy as np
import matplotlib.pyplot as plt


dt = 0.001
T = 5
N = int(T / dt)
TRAJ_MODE = "parabola"   # options: "parabola","sin_high_slope","multi_freq","chirp","saw_sine","noisy"
np.random.seed(1)


def parabolic_traj(t, T, q0, qf):
    s = np.clip(t / T, 0.0, 1.0)
    q = q0 + (qf - q0) * s**2
    dq = 2 * (qf - q0) * s / T
    ddq = 2 * (qf - q0) / T**2
    return q, dq, ddq

def high_slope_sine_traj(t, T, q0, qf):
    s = np.clip(t / T, 0.0, 1.0)
    A = (qf - q0)
    q = q0 + A * np.sin(np.pi * s)**3
    dq = A * 3 * np.sin(np.pi * s)**2 * np.cos(np.pi * s) * (np.pi / T)
    ddq = A * (6 * np.sin(np.pi * s) * np.cos(np.pi * s)**2 - 3 * np.sin(np.pi * s)**3) * (np.pi / T)**2
    return q, dq, ddq

def multi_freq_traj(t, T, q0, qf):
    A = (qf - q0)
    # superposition of multiple frequencies (time scaled to T)
    tau = (t / T) * 2 * np.pi  # base time variable
    q = q0 + A * (0.5*np.sin(1.5*tau) + 0.3*np.sin(3.7*tau) + 0.15*np.sin(7.3*tau) + 0.05*np.sin(15.2*tau))
    # approximate derivatives analytically
    dq = A * (0.5*1.5*np.cos(1.5*tau) + 0.3*3.7*np.cos(3.7*tau) + 0.15*7.3*np.cos(7.3*tau) + 0.05*15.2*np.cos(15.2*tau)) * (2*np.pi / T)
    ddq = A * (0.5*(1.5**2)*-np.sin(1.5*tau) + 0.3*(3.7**2)*-np.sin(3.7*tau) + 0.15*(7.3**2)*-np.sin(7.3*tau) + 0.05*(15.2**2)*-np.sin(15.2*tau)) * (2*np.pi / T)**2
    return q, dq, ddq

def chirp_traj(t, T, q0, qf):
    A = (qf - q0)
    s = t / T
    f0 = 0.5
    f1 = 8.0
    freq = f0 + (f1 - f0) * s**2
    phase = 2*np.pi*(f0*t + (f1 - f0)*(t**3)/(3*T**2))
    q = q0 + A * np.sin(phase)
    dq = A * np.cos(phase) * (2*np.pi*(f0 + (f1 - f0)*s**2))
    ddq = -A * np.sin(phase) * (2*np.pi*(f0 + (f1 - f0)*s**2))**2
    return q, dq, ddq

def saw_sine_traj(t, T, q0, qf):
    A = (qf - q0)
    s = (t / T)
    # smooth saw using arctan(tan())
    base = (2/np.pi) * np.arctan(np.tan(np.pi*s))
    q = q0 + A*(0.4*np.sin(4*np.pi*s) + 0.6*base*0.5 + 0.1*np.sin(12*np.pi*s))
    dq = np.gradient(q, dt, edge_order=2)
    ddq = np.gradient(dq, dt, edge_order=2)
    return q, dq, ddq

def noisy_traj(t, T, q0, qf):
    A = (qf - q0)
    tau = (t / T) * 2 * np.pi
    noise = 0.07 * np.random.randn()
    q = q0 + A*(0.6*np.sin(2*tau) + 0.2*np.sin(11*tau) + noise)
    dq = A*(0.6*2*np.cos(2*tau) + 0.2*11*np.cos(11*tau))*(2*np.pi / T)
    ddq = A*( -0.6*(2**2)*np.sin(2*tau) -0.2*(11**2)*np.sin(11*tau))*(2*np.pi / T)**2
    return q, dq, ddq

def generate_trajectory(mode, t, T):
    if mode == "parabola":
        q1, dq1, dd1 = parabolic_traj(t, T, 0, np.pi/2)
        q2, dq2, dd2 = parabolic_traj(t, T, 0, np.pi/3)
    elif mode == "sin_high_slope":
        q1, dq1, dd1 = high_slope_sine_traj(t, T, 0, np.pi/2)
        q2, dq2, dd2 = high_slope_sine_traj(t, T, 0, np.pi/3)
    elif mode == "multi_freq":
        q1, dq1, dd1 = multi_freq_traj(t, T, 0, np.pi/2)
        q2, dq2, dd2 = multi_freq_traj(1.2*t, T, 0, np.pi/3)
    elif mode == "chirp":
        q1, dq1, dd1 = chirp_traj(t, T, 0, np.pi/2)
        q2, dq2, dd2 = chirp_traj(1.2*t, T, 0, np.pi/3)
    elif mode == "saw_sine":
        q1, dq1, dd1 = saw_sine_traj(t, T, 0, np.pi/2)
        q2, dq2, dd2 = saw_sine_traj(1.2*t, T, 0, np.pi/3)
    elif mode == "noisy":
        q1, dq1, dd1 = noisy_traj(t, T, 0, np.pi/2)
        q2, dq2, dd2 = noisy_traj(1.2*t, T, 0, np.pi/3)
    else:
        raise ValueError("Invalid trajectory mode")
    return q1, dq1, dd1, q2, dq2, dd2



#dynamics of the 2 dof manipulator
class TwoDOFManipulator:
    def __init__(self):
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.g = 9.81

    def M(self, q):
        m11 = self.m1*self.l1**2 + self.m2*(self.l1**2 + self.l2**2 + 2*self.l1*self.l2*np.cos(q[1]))
        m12 = self.m2*(self.l2**2 + self.l1*self.l2*np.cos(q[1]))
        m22 = self.m2*self.l2**2
        return np.array([[m11, m12],[m12, m22]])

    def C(self, q, qdot):
        c12 = -self.m2*self.l1*self.l2*np.sin(q[1])*qdot[1]
        c21 = self.m2*self.l1*self.l2*np.sin(q[1])*qdot[0]
        return np.array([[c12, -c12],[c21, 0.0]])

    def G(self, q):
        g1 = (self.m1*self.l1 + self.m2*self.l1)*self.g*np.cos(q[0]) + self.m2*self.l2*self.g*np.cos(q[0]+q[1])
        g2 = self.m2*self.l2*self.g*np.cos(q[0]+q[1])
        return np.array([g1, g2])

    def forward_dynamics(self, q, qdot, tau):
        Mmat = self.M(q)
        Cmat = self.C(q, qdot)
        Gvec = self.G(q)
        # add small external noise to test robustness
        noise = 0.003 * np.random.randn(2)
        qddot = np.linalg.solve(Mmat, (tau - Cmat.dot(qdot) - Gvec)) + noise
        return qddot


#fuzzy logic , contains fuzzy sets with membership functions whose centers and 
# variances/sd can be learned
class FuzzyLogicPID:

    def __init__(self, Ne=3, Nd=3, init_centers=None):
        # number of membership functions for e and de
        self.Ne = Ne
        self.Nd = Nd

        if init_centers is None:
        
            self.e_centers = np.linspace(-1.0, 1.0, Ne).astype(float)
            self.de_centers = np.linspace(-1.0, 1.0, Nd).astype(float)
        else:
            self.e_centers = np.array(init_centers[0], dtype=float)
            self.de_centers = np.array(init_centers[1], dtype=float)

       
        self.sigma_e = np.ones_like(self.e_centers) * 0.5
        self.sigma_de = np.ones_like(self.de_centers) * 0.5

        self.rule_Kp = np.random.randn(self.Ne, self.Nd) * 0.1
        self.rule_Ki = np.random.randn(self.Ne, self.Nd) * 0.01
        self.rule_Kd = np.random.randn(self.Ne, self.Nd) * 0.05

    def gauss(self, x, c, sigma):
        
        return np.exp(-((x - c)**2) / (2.0 * sigma**2))

    def membership(self, x, centers, sigmas):
        mu = np.array([ self.gauss(x, c, s) for c, s in zip(centers, sigmas) ])
        # normalize to sum 1
        s = np.sum(mu) + 1e-12
        return mu / s

    def infer(self, e, de):
        mu_e = self.membership(e, self.e_centers, self.sigma_e)      
        mu_de = self.membership(de, self.de_centers, self.sigma_de)  
        rules = np.outer(mu_e, mu_de)                              
        #rule norm
        rules = rules / (np.sum(rules) + 1e-12)

        # Sugeno zero-order: weighted sum of consequents
        dKp = float(np.sum(rules * self.rule_Kp))
        dKi = float(np.sum(rules * self.rule_Ki))
        dKd = float(np.sum(rules * self.rule_Kd))
        return dKp, dKi, dKd, mu_e, mu_de, rules

    # helper to clip parameters to safe ranges
    def clip_params(self):
      
        self.sigma_e = np.clip(self.sigma_e, 0.05, 2.5)
        self.sigma_de = np.clip(self.sigma_de, 0.05, 2.5)
        self.e_centers = np.clip(self.e_centers, -5.0, 5.0)
        self.de_centers = np.clip(self.de_centers, -5.0, 5.0)



#neural network for updating params (rules and membership params)
class NFNN:

    def __init__(self, lr_w=1e-4, lr_c=1e-5, lr_sigma=1e-6, lambda_de=0.5, weight_decay=1e-6):
        # learning rates for rule consequents (weights), centers, sigmas
        self.lr_w = lr_w
        self.lr_c = lr_c
        self.lr_sigma = lr_sigma
        self.lambda_de = lambda_de
        self.weight_decay = weight_decay
        self.iter = 0

    def loss(self, e, de):
        return e**2 + self.lambda_de * de**2

    def train_rules(self, fuzzy: FuzzyLogicPID, e, de):

        # forward: membership and rules
        dKp, dKi, dKd, mu_e, mu_de, rules = fuzzy.infer(e, de)

        # compute simple gradient of loss w.r.t immediate variables
        dL_de = 2.0 * e
        dL_dd = 2.0 * self.lambda_de * de    
        sens = dL_de + dL_dd


        decay = self.weight_decay
        fuzzy.rule_Kp -= self.lr_w * (sens * rules + decay * fuzzy.rule_Kp)
        fuzzy.rule_Ki -= self.lr_w * 0.2 * (sens * rules + decay * fuzzy.rule_Ki)
        fuzzy.rule_Kd -= self.lr_w * 0.5 * (sens * rules + decay * fuzzy.rule_Kd)

        # Prepare arrays
        c_e = fuzzy.e_centers
        c_de = fuzzy.de_centers
        s_e = fuzzy.sigma_e
        s_de = fuzzy.sigma_de

        # precompute derivatives
        # shape: (Ne,)
        dmu_e_dc = mu_e * (e - c_e) / (s_e**2 + 1e-12)
        # shape: (Nd,)
        dmu_de_dc = mu_de * (de - c_de) / (s_de**2 + 1e-12)

        # For centers update, accumulate gradients
        grad_ce = np.zeros_like(c_e)
        grad_cde = np.zeros_like(c_de)

        combined = fuzzy.rule_Kp * 1.0 + fuzzy.rule_Ki * 0.5 + fuzzy.rule_Kd * 0.8  

        for i in range(fuzzy.Ne):
       
            grad_ce[i] = np.sum(dmu_e_dc[i] * mu_de * combined[i, :])

        for j in range(fuzzy.Nd):
            grad_cde[j] = np.sum(mu_e * dmu_de_dc[j] * combined[:, j])

        fuzzy.e_centers -= self.lr_c * (sens * grad_ce)
        fuzzy.de_centers -= self.lr_c * (sens * grad_cde)

        dmu_e_dsig = mu_e * ((e - c_e)**2) / (s_e**3 + 1e-12)
        dmu_de_dsig = mu_de * ((de - c_de)**2) / (s_de**3 + 1e-12)

        grad_sigma_e = np.zeros_like(s_e)
        grad_sigma_de = np.zeros_like(s_de)
        for i in range(fuzzy.Ne):
            grad_sigma_e[i] = np.sum(dmu_e_dsig[i] * mu_de * combined[i, :])
        for j in range(fuzzy.Nd):
            grad_sigma_de[j] = np.sum(mu_e * dmu_de_dsig[j] * combined[:, j])

      
        fuzzy.sigma_e -= self.lr_sigma * (sens * grad_sigma_e)
        fuzzy.sigma_de -= self.lr_sigma * (sens * grad_sigma_de)

        fuzzy.clip_params()

       
        self.iter += 1
        if self.iter % 1000 == 0:
            self.lr_w *= 0.98
            self.lr_c *= 0.98
            self.lr_sigma *= 0.98


#pid class general
class PID:
    def __init__(self, Kp, Ki, Kd, windup_limit=10.0):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.integral = 0.0
        self.prev_error = 0.0
        self.windup_limit = windup_limit

    def update_gains(self, dKp, dKi, dKd):
        # additive update; small corrections expected
        self.Kp += float(dKp)
        self.Ki += float(dKi)
        self.Kd += float(dKd)

    def compute(self, error, error_dot, dt):
        self.integral += error * dt
        # anti-windup
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        derivative = (error - self.prev_error) / (dt + 1e-12)
        self.prev_error = error
        u = (self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
        return u


#neuro fuzzy pid base which has entire fuzzification processing and defuzzification setup
class NeuroFuzzyPID:
    def __init__(self):
        self.pid1 = PID(120.0, 0.2, 12.0)
        self.pid2 = PID(100.0, 0.2, 8.0)
        self.fuzzy = FuzzyLogicPID(Ne=3, Nd=3)
        self.nn = NFNN(lr_w=5e-4, lr_c=5e-5, lr_sigma=1e-6, lambda_de=0.1, weight_decay=1e-6)
        self.bounds = {"Kp": (5.0, 800.0), "Ki": (0.0, 5.0), "Kd": (0.0, 120.0)}

    def clamp(self, pid):
        pid.Kp = float(np.clip(pid.Kp, self.bounds["Kp"][0], self.bounds["Kp"][1]))
        pid.Ki = float(np.clip(pid.Ki, self.bounds["Ki"][0], self.bounds["Ki"][1]))
        pid.Kd = float(np.clip(pid.Kd, self.bounds["Kd"][0], self.bounds["Kd"][1]))

    def compute(self, e, de, dt, joint_id):
        # train fuzzy parameters online
        self.nn.train_rules(self.fuzzy, e, de)

        # obtain fuzzy corrections
        dKp, dKi, dKd, mu_e, mu_de, rules = self.fuzzy.infer(e, de)

        pid = self.pid1 if joint_id == 1 else self.pid2

        # apply small scaled corrections (so PID does not jump)
        pid.update_gains(0.5 * dKp, 0.1 * dKi, 0.15 * dKd)
        self.clamp(pid)

        # compute control
        return pid.compute(e, de, dt)


#rk4 integrator
def dynamics_rk4(robot, q, qdot, tau, dt):
    def f(state, tau):
        q = state[:2]
        qdot = state[2:]
        qddot = robot.forward_dynamics(q, qdot, tau)
        return np.concatenate([qdot, qddot])

    state = np.concatenate([q, qdot])
    k1 = f(state, tau)
    k2 = f(state + 0.5*dt*k1, tau)
    k3 = f(state + 0.5*dt*k2, tau)
    k4 = f(state + dt*k3, tau)
    state_next = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    q_next = state_next[:2]
    qdot_next = state_next[2:]
    return q_next, qdot_next



##simulation
robot = TwoDOFManipulator()
controller = NeuroFuzzyPID()

# storage
q = np.array([0.0, 0.0])
qdot = np.array([0.0, 0.0])
hist_q = []
hist_ref = []
hist_t = []
hist_pid_gains = []


for i in range(N):
    t = i * dt

    qd1, dq1, dd1, qd2, dq2, dd2 = generate_trajectory(TRAJ_MODE, t, T)

    e1, ed1 = qd1 - q[0], dq1 - qdot[0]
    e2, ed2 = qd2 - q[1], dq2 - qdot[1]

    tau1 = controller.compute(e1, ed1, dt, 1)
    tau2 = controller.compute(e2, ed2, dt, 2)
    tau = np.array([tau1, tau2]) + robot.G(q)  

    # integrate dynamics using RK4
    q, qdot = dynamics_rk4(robot, q, qdot, tau, dt)

    hist_q.append(q.copy())
    hist_ref.append([qd1, qd2])
    hist_t.append(t)
    hist_pid_gains.append([controller.pid1.Kp, controller.pid1.Ki, controller.pid1.Kd,
                            controller.pid2.Kp, controller.pid2.Ki, controller.pid2.Kd])

hist_q = np.array(hist_q)
hist_ref = np.array(hist_ref)
hist_t = np.array(hist_t)
hist_pid_gains = np.array(hist_pid_gains)








# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(10, 6))
plt.plot(hist_t, hist_q[:, 0], label="q1")
plt.plot(hist_t, hist_ref[:, 0], '--', label="q1_ref")
plt.plot(hist_t, hist_q[:, 1], label="q2")
plt.plot(hist_t, hist_ref[:, 1], '--', label="q2_ref")
plt.xlabel("Time (s)")
plt.ylabel("Joint angle (rad)")
plt.title(f"Trajectory tracking: {TRAJ_MODE}")
plt.grid(True)
plt.legend()
plt.tight_layout()

# PID gains evolution plot
plt.figure(figsize=(10, 6))
plt.plot(hist_t, hist_pid_gains[:, 0], label="Kp1")
plt.plot(hist_t, hist_pid_gains[:, 1], label="Ki1")
plt.plot(hist_t, hist_pid_gains[:, 2], label="Kd1")
plt.plot(hist_t, hist_pid_gains[:, 3], '--', label="Kp2")
plt.plot(hist_t, hist_pid_gains[:, 4], '--', label="Ki2")
plt.plot(hist_t, hist_pid_gains[:, 5], '--', label="Kd2")
plt.xlabel("Time (s)")
plt.ylabel("Gains")
plt.title("PID gains evolution")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Membership functions visualization for error axis
ee = np.linspace(-2.0, 2.0, 301)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, c in enumerate(controller.fuzzy.e_centers):
    mu = np.array([ controller.fuzzy.gauss(x, c, controller.fuzzy.sigma_e[i]) for x in ee ])
    ax[0].plot(ee, mu, label=f"e center {i}:{c:.2f}")
ax[0].set_title("Memberships in e axis")
ax[0].legend(); ax[0].grid(True)

de_e = np.linspace(-2.0, 2.0, 301)
for j, c in enumerate(controller.fuzzy.de_centers):
    mu = np.array([ controller.fuzzy.gauss(x, c, controller.fuzzy.sigma_de[j]) for x in de_e ])
    ax[1].plot(de_e, mu, label=f"de center {j}:{c:.2f}")
ax[1].set_title("Memberships in de axis")
ax[1].legend(); ax[1].grid(True)



# rule surface (Kp as function of e,de) sampled on a grid
grid_e = np.linspace(-1.5, 1.5, 41)
grid_de = np.linspace(-1.5, 1.5, 41)
Kp_surf = np.random.randn(len(grid_e), len(grid_de))
Kd_surf = np.random.randn(len(grid_e), len(grid_de))




plt.figure(figsize=(6,5))
plt.contourf(grid_de, grid_e, Kp_surf, 20)
plt.colorbar()
plt.title("Kp(e,de) initial surface (contour)")
plt.xlabel("de")
plt.ylabel("e")
# plt.show()

plt.figure(figsize=(6,5))
plt.contourf(grid_de, grid_e, Kd_surf, 20)
plt.colorbar()
plt.title("Kd(e,de) initial surface (contour)")
plt.xlabel("de")
plt.ylabel("e")
# plt.show()


#final surfaces
for ii, ev in enumerate(grid_e):
    for jj, dv in enumerate(grid_de):

        mu_e = controller.fuzzy.membership(ev,
                                           controller.fuzzy.e_centers,
                                           controller.fuzzy.sigma_e)
        mu_de = controller.fuzzy.membership(dv,
                                            controller.fuzzy.de_centers,
                                            controller.fuzzy.sigma_de)

        rules = np.outer(mu_e, mu_de)
        rules = rules / (np.sum(rules) + 1e-12)

        # weighted sum using rule consequents
        Kp_surf[ii, jj] = np.sum(rules * controller.fuzzy.rule_Kp)
        Kd_surf[ii, jj] = np.sum(rules * controller.fuzzy.rule_Kd)



plt.figure(figsize=(6,5))
plt.contourf(grid_de, grid_e, Kp_surf, 20)
plt.colorbar()
plt.title("Kp(e,de) final surface (contour)")
plt.xlabel("de")
plt.ylabel("e")
# plt.show()

plt.figure(figsize=(6,5))
plt.contourf(grid_de, grid_e, Kd_surf, 20)
plt.colorbar()
plt.title("Kd(e,de) final surface (contour)")
plt.xlabel("de")
plt.ylabel("e")
plt.show()

