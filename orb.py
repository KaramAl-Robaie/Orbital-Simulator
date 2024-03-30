import sympy as sp
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons 
from astropy.time import Time

ids = {"Jupiter": 599, 
       "Io": 501,
       "Europa": 502,
       "Ganymede": 503,
       "Callisto": 504,}

masses = {"Jupiter": 1.898e27, 
          "Io": 8.93e22,
          "Europa": 4.80e22,
          "Ganymede": 1.482e23,
          "Callisto": 1.076e23,}

colours= {"Jupiter": "red", 
          "Io": "orange",
          "Europa": "yellow",
          "Ganymede": "green",
          "Callisto": "blue"}

bodies = list(ids.keys())
N = len(bodies)


pv = sp.symbols(f"p1:{3*N+1}", real=True) # Vector of momenta
qv = sp.symbols(f"q1:{3*N+1}", real=True) # Vector of positions

# Massage these scalars into a nicer vector format
momenta = {}
positions = {}
for i, body in enumerate(bodies):
    momenta[body] = sp.Matrix([pv[3*i+0], pv[3*i+1], pv[3*i+2]])
    positions[body] = sp.Matrix([qv[3*i+0], qv[3*i+1], qv[3*i+2]])

G = 1.4878e-34

# Construct kinetic energy
V=0
for body in bodies:
    p = momenta[body]
    m = masses[body]
    V += p.dot(p)/(2*m)

# Construct potential energy
T=0
for i, body_a in enumerate(bodies):
    m_a = masses[body_a]
    q_a = positions[body_a]

    for body_b in bodies[i+1:]: 
        m_b = masses[body_b] 
        q_b = positions[body_b]
        
        T -= G * m_a * m_b / (q_a - q_b).norm()

# Write the Hamiltonian
H=V+T

# Derive equations of motion
equations = [-sp.diff(H, q) for q in qv] + [+sp.diff(H, p) for p in pv]

# Lambdify to construct right-hand side of ODE
t = sp.Symbol("t", real=True, nonnegative=True) 
f = sp.lambdify([t, pv + qv], equations)


ic = np.zeros(6*N)

# Start time of our simulation
epoch = Time('2023-01-01 00:00:00').jd

for i, body in enumerate(bodies):
    # Query Horizons database for coordinates relative to
    # barycentre of the solar system
    query = Horizons(id=ids[body], location='500@0', epochs=epoch) 
    vec = query.vectors()

    # Multiply by mass to compute momentum
    ic[3*i+0] = vec['vx'][0]*masses[body]
    ic[3*i+1] = vec['vy'][0]*masses[body]
    ic[3*i+2] = vec['vz'][0]*masses[body]
    ic[3*i+3*N+0] = vec['x'][0]
    ic[3*i+3*N+1] = vec['y'][0]
    ic[3*i+3*N+2] = vec['z'][0]

    # Plot initial position of the planet in grey
    plt.plot(vec['x'][0], vec['y'][0], 'o', color=colours[body])

D = 170 # Equivalent to 10 Castillo Orbits
t0, t1 = 0, D
t_eval = np.linspace(t0, t1, t1+1) # store simulation data every day

trajectory = sc.integrate.solve_ivp(f, (t0, t1), ic, t_eval=t_eval,
                             method='DOP853', rtol=1e-8, atol=1e-8)


for (i, body) in enumerate(bodies):
    x = trajectory.y[3*N + 3*i + 0, :] 
    y = trajectory.y[3*N + 3*i + 1, :]

    # Plot trajectory
    plt.plot(x, y, label=body, color=colours[body]) # Plot final position of the planet
    plt.plot(x[-1], y[-1], 'o', color=colours[body]) # Annotate final position with letter
    
plt.gcf().set_dpi(300)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xlabel(r"$x$ [AU]")
plt.ylabel(r"$y$ [AU]")
plt.gca().set_aspect('equal')
plt.show()


###############
###  Finding the orbital periods
###############


# Creating a function that returns the times 
# when the y co-ordinate of Jupiter and the moon are equal
pqv = pv + qv

def orbits(body_index):
    def g(t, pqv):
        return pqv[3*N + 3*body_index + 1] - pqv[3*N + 1]

    endpoints = sc.integrate.solve_ivp(f, (t0, t1), ic, t_eval=t_eval,
                             method='DOP853', events=(g,))
    
    return endpoints.t_events[0]


# Calling the function on each moon, and storing the first time,
# last time and the no. of orbits
orbit_lengths = {}
for j in np.arange(1, N):
    data = orbits(j)
    start = data[0]
    end = data[-1]
    no_orbits = (len(data) - 1) / 2 # The y co-ordinates are equal every half orbit

    orbit_lengths[bodies[j]] = (end - start) / (no_orbits) # Taking the mean orbit_length accross all 170 days


# Printing out the results
for body in orbit_lengths:
    print(f"The avg orbit of {body} is {orbit_lengths[body]:.2f} days...")
    print(f"...and is {orbit_lengths[body]/orbit_lengths['Io']:.2f} longer than Io's orbit.\n")
