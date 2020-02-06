import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Hodgkin_Huxley:


    def __init__(self):
        # Membrane capacitance and potential
        self.C_m = 1.0
        self.V_m = -70.0

        # Max sodium conductance and membrane potential
        self.g_Na = 120.0
        self.V_Na = 56.0

        # Max potassium conductance and membrane potential
        self.g_K = 36.0
        self.V_K = -77.0

        # Max leak conductance and membrane potential
        self.g_l = 0.3
        self.V_l = -60

        # Time to integrate over
        self.time = np.arange(0.0, 1000.0, 0.1)

        # ion variables
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32

    def I_inj(self, t):
        return np.piecewise(t, [(t < 500) * (t >= 200), (t < 950) * (t >= 700)], [10, 50])
        #return np.piecewise(self.time, [(self.time< 500)*(self.time>=200), (self.time< 950)*(self.time>=700)], [10,50])

    # potassium rate functions
    def a_n(self):
        return 0.01 * (self.V_m + 55) / (1 - np.exp(-(self.V_m + 55.0) / 10)) # + 1.E-7)

    def B_n(self):
        return 0.125 * np.exp(-(self.V_m + 65.0) / 80)

    # sodium rate functions
    def a_m(self):
        return 0.1 * (40 + self.V_m) / (1.0 - np.exp(-(self.V_m + 40.0) / 10.0)) # + 1.E-7)

    def a_h(self):
        return 0.07 * np.exp(-(self.V_m + 65.0) / 20.0)

    def B_m(self):
        return 4 * np.exp(-(self.V_m + 65.0) / 18.0)

    def B_h(self):
        return 1 / (np.exp(-(35 + self.V_m) / 10) + 1)

    # current functions
    def K_current(self):
        return self.g_K * self.n**4 * (self.V_m - self.V_K)

    def Na_current(self):
        return self.g_Na * self.m**3 * self.h * (self.V_m - self.V_Na)

    def leak_current(self):
        return self.g_l * (self.V_m - self.V_l)

    def hh_model(self, y, t, parameters):
        '''Takes in y dynamics variables, V_m, n, m and h. Array of time steps to integrate over.
        List of constant parameters, reversal potentials and conductances

        Returns values for derivatives w.r.t time for Voltage and ion rate coefficience n, m and h.
        '''
        self.V_m, self.n, self.m, self.h = y
        self.g_K, self.V_K, self.g_Na, self.V_Na, self.g_l, self.C_m = parameters

        # Total current through the membrane
        dVdt = self.I_inj(t) - self.K_current() - self.Na_current() - self.leak_current() / self.C_m

        # Derivative of n, potassium channel activation, w.r.t. time
        dndt = self.a_n() * (1 - self.n) - self.B_n() * self.n

        # Derivative of m, sodium channel activion, w.r.t. time
        dmdt = self.a_m() * (1 - self.m) - self.B_m() * self.m

        # Derivative of h, sodium channel in-activion, w.r.t. time

        dhdt = self.a_h() * (1 - self.h) - self.B_h() * self.h

        return [dVdt, dndt, dmdt, dhdt]

    def simulate(self):

        y0 = [self.V_m, self.n, self.m, self.h]
        parameters = [self.g_K, self.V_K, self.g_Na, self.V_Na, self.g_l, self.C_m]

        sol = odeint(self.hh_model, y0, self.time, args=(parameters,))
        return sol

model = Hodgkin_Huxley()
model.simulate()
#print(model.sol)