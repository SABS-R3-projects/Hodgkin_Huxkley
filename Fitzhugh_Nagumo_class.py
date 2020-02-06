import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate


class Fitzhugh_Nagumo(object):
    ''' Class inlcuding model and simulation of Fitzhugh_Nagumo model
    '''

    def __init__(self):
        self.I_inj = 0.34
        self.a = 0.7
        self.b = 0.8
        self.c = 10

        # state variables
        # membrane potential
        self.V_m = 2.0

        # recovery constant
        self.w = 1.0

        self.t_0 = np.arange(0.0, 20, 0.01)

        # time to simulate over
        self.t_0 = np.arange(0.0, 5, 0.01)
    pass

    def Fitzhugh_Nagumo_model(self, y0, t, parameters):
        """Fitzhugh_Nagumo model
        V = membrane voltage
        w = recovery variable
        """

        self.V_m, self.w = y0
        self.a, self.b, self.c = parameters

        dvdt = self.c * (self.V_m - self.w - np.power(self.V_m,3)/3 + self.I_inj)

        dwdt = self.V_m + self.a - self.b * self.w

        return [dvdt, dwdt]

    def simulate(self):
        """Solve the Fitzhugh-Nagumo model

        """

        y0 = [self.V_m, self.w]
        parameters = [self.a, self.b, self.c]

        sol = odeint(self.Fitzhugh_Nagumo_model, y0, self.time, args=(parameters,))

        return sol



I = 0.34 #external stimulus
a = 0.7
b = 0.8
c = 10

def FHN(state, t):
    """
    FitzHugh-Nagumo Equations
    u : the membrane potential
    v : a recovery variable
    """
    u, v = state
    dot_u = c * (-v + u - pow(u,3)/3 + I)
    dot_v = u - b * v + a
    return dot_u, dot_v

#initial state
u0 = 2.0
v0 = 1.0

t = np.arange(0.0, 5, 0.01)


t0 = np.arange(0.0, 20, 0.01)
y_all = integrate.odeint(FHN, [u0, v0], t0)
u_all = y_all[:,0]
v_all = y_all[:,1]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
fig.suptitle("FitzHugh-Nagumo Model")

len_t = len(t)
dt = 5 #time steps

# 1step
def update(i):
    global y, y0

    # initial y0
    if i ==0:
        y0 = [u0, v0]

    # Delete Graph
    ax1.cla()
    ax2.cla()

    # Solve ODE
    y = integrate.odeint(FHN, y0, t)

    # Update y0
    y0 = (y[dt,0], y[dt,1])

    # get u and v
    u = y[:,0]
    v = y[:,1]

    #Phase Space
    ax1.plot(u_all, v_all, color="k", dashes=[1, 6])
    ax1.plot(u[len_t-20:len_t-1], v[len_t-20:len_t-1],color="r")
    ax1.plot(u[len_t-1],v[len_t-1],'o--', color="r") #uのmarker
    ax1.set_xlabel("u : membrane potential / Volt")
    ax1.set_ylabel("v : recovery variable")
    ax1.set_xlim([-2.2,2.2])
    ax1.set_ylim([-0.5,1.5])
    ax1.set_title("Phase Space")
    ax1.grid()

    #Membrane Potential
    ax2.plot(t, u, label="u : membrane potential", color="#ff7f0e")
    ax2.plot(t, v, label="v : recovery variable", color="#1f77b4")
    ax2.plot(t[len_t-1], u[len_t-1],'o--', color="#ff7f0e") #uのmarker
    ax2.plot(t[len_t-1], v[len_t-1],'o--', color="#1f77b4") #vのmarker
    ax2.set_title("Membrane Potential / Volt")
    ax2.set_ylim([-2.2,2.0])
    ax2.grid()
    ax2.legend(bbox_to_anchor=(0, 1),
               loc='upper left',
               borderaxespad=0)

ani = animation.FuncAnimation(fig, update, interval=100,
                              frames=300)
plt.show()
#ani.save("FitzHugh-Nagumo_all.mp4") #save