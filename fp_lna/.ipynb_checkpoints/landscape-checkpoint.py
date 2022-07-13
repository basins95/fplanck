import numpy as np
# import numba ?
from scipy import interpolate
from scipy import optimize
from numpy import linalg as LA
import matplotlib.pyplot as plt

import scipy

# for the EuMa sim
from scipy.stats import multivariate_normal
# for the LNA
from scipy.integrate import solve_ivp


from fp_lna.functions import ffix_solve, point_type, get_x_initial, string1_adapted


class landscape:
    def __init__(self, potential_V, Gradient_G, G_point, Jacobian_J, uvw, lims, 
                 tolerance = 1e-5, iterations = 400):
        """
        Solve the Fokker-Planck equation
        Arguments:
            potential
            uvw             parameters needed for the distribution
            
        Optional pars:
            tolerance
            iterations
            (epsilon)
        """
        
        self.V = potential_V
        self.G = Gradient_G
        self.G_point = G_point
        self.J = Jacobian_J
        self.uvw = uvw
        self.lims = lims
        self.tolerance = tolerance
        self.iterations = iterations
            
		# automatically returns fixed points
        fixed_points = ffix_solve(self.uvw, self.G_point)
        self.fixed_points = fixed_points
        saddle_attractors = point_type(self.G, self.J, fixed_points, self.uvw)
        
        the_saddle = fixed_points[saddle_attractors==1,:][0]
        self.xs = the_saddle
        attractors = fixed_points[saddle_attractors==0,:]
        self.xa = attractors[0,:]
        self.xb = attractors[1,:] 
		
		# obtaining manifolds
        stable_unstable = np.array([False, True])
        Nt = 2**6
        s = np.linspace(0,1,Nt)
        
        for stable in stable_unstable:
            if stable:
                # if we want the stable, we set integration to backward
                forward = False
                x_initial = get_x_initial(self.xs, self.xa, self.xb, stable, self.J, self.uvw, s)
                manifold = string1_adapted(x_initial, self.G, self.uvw, forward, 
                                           self.lims, self.tolerance, self.iterations)
                self.x_stable = manifold
            else:
                forward = True
                x_initial = get_x_initial(self.xs, self.xa, self.xb, stable, self.J, self.uvw, s)
                manifold = string1_adapted(x_initial, self.G, self.uvw, forward, 
                                           self.lims, self.tolerance, self.iterations)
                self.x_unstable = manifold

		
# 	def propagate_interval(self, initial, tf, Nsteps=None, dt=None, normalize=True):
#         """Propagate an initial probability distribution over a time interval, return time and the probability distribution at each time-step
#         Arguments:
#             initial      initial probability density function
#             tf           stop time (inclusive)
#             Nsteps       number of time-steps (specifiy Nsteps or dt)
#             dt           length of time-steps (specifiy Nsteps or dt)
#             normalize    if True, normalize the initial probability
#         """
#         p0 = initial(*self.grid)
#         if normalize:
#             p0 /= np.sum(p0)

#         if Nsteps is not None:
#             dt = tf/Nsteps
#         elif dt is not None:
#             Nsteps = np.ceil(tf/dt).astype(int)
#         else:
#             raise ValueError('specifiy either Nsteps or Nsteps')

#         time = np.linspace(0, tf, Nsteps)
#         pf = expm_multiply(self.master_matrix, p0.flatten(), start=0, stop=tf, num=Nsteps, endpoint=True)
#         return time, pf.reshape((pf.shape[0],) + tuple(self.Ngrid))

#     def probability_current(self, pdf):
#         """Obtain the probability current of the given probability distribution"""
#         J = np.zeros_like(self.force_values)
#         for i in range(self.ndim):
#             J[i] = -(self.diffusion[i]*np.gradient(pdf, self.resolution[i], axis=i) 
#                   - self.mobility[i]*self.force_values[i]*pdf)

#         return J