"""
Crank-Nicolson solver for mixed EB system in 3D equation with Dirichlet boundary conditions:

  E_t = - curl B + f
  B_t = + curl E
  Exn = g on boundary
  B.n = h on boundary
  E(0) = E0, B(0) = B0

Crank-Nicolson time stepping:
(u_{n+1} - u_n)/k = (1/2) ( CC u_{n+1} + f_{n+1} + CC u_n + f_n )
where u = (E) and CC = ( 0 -curl ) and f = (f)
          (B)          ( curl  0 )         (0)
"""

from __future__ import division
from math import floor
from collections import OrderedDict
from dolfin import *
import helper_module as helper
from helper_module import projection

info('Crank-Nicolson for mixed EB system in 3D with tangential BCs')

# boundary data, initial data, forcing function, initial time, stopping time, time step
# note: g and f may depend on t

example = 'z-direction plane wave'
# example = 'periodic z-direction plane wave'
# example = 'kasner'

if example == 'z-direction plane wave':
    """
    Gravitational wave in the z-direction
    """

    t0 = 0.0
    # T = 2.5
    T = 7.5
    periodicbc = False

    omega = 10
    Ap = 2.
    Ac = .5
    # omega = 20
    # Ap = 1.
    # Ac = 0.

    j = sigmaex = Expression(('0.','0.','0.'), cell=triangle, t=t0)
    g = Eex = Expression( (('-Ap*sin(omega*(x[2]-t))', '-Ac*sin(omega*(x[2]-t))', '0.'),
                            ( '-Ac*sin(omega*(x[2]-t))', 'Ap*sin(omega*(x[2]-t))', '0.'),
                            ( '0.', '0.', '0.')), t=t0,omega=omega,Ap=Ap,Ac=Ac)
    h = Bex = Expression( (('-Ac*sin(omega*(x[2]-t))', 'Ap*sin(omega*(x[2]-t))', '0.'),
                            ( 'Ap*sin(omega*(x[2]-t))', 'Ac*sin(omega*(x[2]-t))', '0.'),
                            ( '0.', '0.', '0.')), t=t0,omega=omega,Ap=Ap,Ac=Ac)
    E0 = Expression( (('-Ap*sin(omega*(x[2]))', '-Ac*sin(omega*(x[2]))', '0.'),
                            ( '-Ac*sin(omega*(x[2]))', 'Ap*sin(omega*(x[2]))', '0.'),
                            ( '0.', '0.', '0.')), omega=omega,Ap=Ap,Ac=Ac)
    B0 = Expression( (('-Ac*sin(omega*(x[2]))', 'Ap*sin(omega*(x[2]))', '0.'),
                            ( 'Ap*sin(omega*(x[2]))', 'Ac*sin(omega*(x[2]))', '0.'),
                            ( '0.', '0.', '0.')), omega=omega,Ap=Ap,Ac=Ac)
    f  = Constant( (('0.', '0.', '0.'),
                    ('0.', '0.', '0.'),
                    ('0.', '0.', '0.')))
elif example == 'periodic z-direction plane wave':
    """
    Gravitational wave in the z-direction
    """

    t0 = 0.0
    T = 2.5
    periodicbc = True

    omega = 6*pi
    Ap = 1.
    Ac = 0.

    # Verify that the wave is in fact periodic
    assert near(omega/pi, int(omega/pi+.5)), 'Wave is not periodic'

    j = sigmaex = Expression(('0.','0.','0.'), cell=triangle, t=t0)
    g = Eex = Expression( (('-Ap*sin(omega*(x[2]-t))', '-Ac*sin(omega*(x[2]-t))', '0.'),
                            ( '-Ac*sin(omega*(x[2]-t))', 'Ap*sin(omega*(x[2]-t))', '0.'),
                            ( '0.', '0.', '0.')), t=t0,omega=omega,Ap=Ap,Ac=Ac)
    h = Bex = Expression( (('-Ac*sin(omega*(x[2]-t))', 'Ap*sin(omega*(x[2]-t))', '0.'),
                            ( 'Ap*sin(omega*(x[2]-t))', 'Ac*sin(omega*(x[2]-t))', '0.'),
                            ( '0.', '0.', '0.')), t=t0,omega=omega,Ap=Ap,Ac=Ac)
    E0 = Expression( (('-Ap*sin(omega*(x[2]))', '-Ac*sin(omega*(x[2]))', '0.'),
                            ( '-Ac*sin(omega*(x[2]))', '-Ap*sin(omega*(x[2]))', '0.'),
                            ( '0.', '0.', '0.')), omega=omega,Ap=Ap,Ac=Ac)
    B0 = Expression( (('-Ac*sin(omega*(x[2]))', 'Ap*sin(omega*(x[2]))', '0.'),
                            ( 'Ap*sin(omega*(x[2]))', 'Ac*sin(omega*(x[2]))', '0.'),
                            ( '0.', '0.', '0.')), omega=omega,Ap=Ap,Ac=Ac)
    f  = Constant( (('0.', '0.', '0.'),
                    ('0.', '0.', '0.'),
                    ('0.', '0.', '0.')))
elif example == 'kasner':
    """
    Kasner cosmology 
    """

    t0 = 1.0
    T = 2.5
    periodicbc = True

    # Expanding flat space
    p1 = 0.
    p2 = 0.
    p3 = 1.

    # Axisymmetric
    # p1 = 2/3
    # p2 = 2/3
    # p3 = -1/3

    # The p's must satisfy the following condition
    assert p1 + p2 + p3 == p1**2 + p2**2 + p3**2 == 1, 'Incorrect choice of p1, p2, and p3.'

    # Take care of the special case when p's are zero.
    if p1 == 0.:
        E11 = '0.'
    else:
        E11 = '-p1*(2*p1-1)*pow(t, p1-1)' 
    if p2 == 0.:
        E22 = '0.'
    else:
        E22 = '-p2*(2*p2-1)*pow(t, p2-1)' 
    if p3 == 0.:
        E33 = '0.'
    else:
        E33 = '-p3*(2*p3-1)*pow(t, p3-1)' 

    j = sigmaex = Expression(('0.','0.','0.'), cell=triangle, t=t0)
    g = Eex = Expression(  (( E11, '0.', '0.' ),
                            ( '0.', E22, '0.' ),
                            ( '0.', '0.', E33 )), t=t0,p1=p1,p2=p2,p3=p3)
    h = Bex = Expression( (( '0.', '0.', '0.' ),
                           ( '0.', '0.', '0.' ),
                           ( '0.', '0.', '0.' )), t=t0)
    f = E0 = B0 = Constant( (( '0.', '0.', '0.' ),
                             ( '0.', '0.', '0.' ),
                             ( '0.', '0.', '0.' )))
else:
    raise 'Incorrect example choice.'

uex = (Eex, Bex)

# Order of method
order_time = 2
order_space = 1
# Save most time steps?
saveframes = False
# Compute the error at most time step?
compute_errors = True
# compute_errors = False
# If not computing errors at most time step, compute error at end? 
compute_errors_at_end = True

class DataList():
    """
    Store errors and constraints through time stepping.
    """

    def __init__(self, V0, W1, W2, V3):
        """
        Attach the spaces needed for constraints and errors.

        Input:  V0 -- space for weak divergence of E
                W1 -- space for errors of E
                W2 -- space for errors of B
                V3 -- space for divergence of B
        """
        self.V0 = V0
        self.W1 = W1
        self.W2 = W2
        self.V3 = V3

        self.solver_E = self.solver_B = self.solver_divE = self.solver_divB = None
        self.data = OrderedDict()

        self.point = (.25,.25,.25)

    def monitor_constraints(self, U, V, solver_div = None):
        """
        Measure divergence, symmetry, and trace.
    
        Input:  U -- Approximate solution
                V -- Space for divergence
        """

        timer = Timer('Computing constraints')
    
        # Verify divergence-free condition
        # (f, solver_div) = projection(div(U), V, solver_div)
        # divfree = sqrt(abs(assemble(inner(f, f)*dx)))
        divfree = sqrt(abs(assemble(inner(div(U), div(U))*dx)))
        info('Constraint: Divergence: {:.2E}'.format(divfree))
    
        # Verify symmetry condition
        from helper_module import vskw
        skwfree = sqrt(abs(assemble(inner(vskw(U), vskw(U))*dx)))
        info('Constraint: Skew: {:.2E}'.format(skwfree))
    
        # Verify tracefree condition
        trfree = sqrt(abs(assemble(inner(tr(U), tr(U))*dx)))
        info('Constraint: Trace: {:.2E}'.format(trfree))
    
        return (divfree, skwfree, trfree, solver_div)

    def update(self, t, ucur, uex):
        """
        Update data at each iterations.
    
        Input:  t -- current time
                ucur -- approximate solution
                uex -- exact solution
        Output: data -- current errors and constraints
        """

        data = self.data
    
        # Split E and B
        (sigmah, Eh, Bh) = ucur.split()
    
        # Update exact solution
        (Eex, Bex) = uex
        for u in uex:
            u.t = t
    
        # Compute error
        (err_E, nor_E, self.solver_E) = helper.monitor_error(Eh, Eex, self.W1, self.solver_E)
        (err_B, nor_B, self.solver_B) = helper.monitor_error(Bh, Bex, self.W2, self.solver_B)
    
        # Compute constraints
        (divfree_E, skwfree_E, trfree_E, self.solver_divE) = self.monitor_constraints(Eh, self.V0, self.solver_divE)
        (divfree_B, skwfree_B, trfree_B, self.solver_divB) = self.monitor_constraints(Bh, self.V3, self.solver_divB)

        # Verify sigma is zero condition
        sigmazero = sqrt(abs(assemble(inner(sigmah, sigmah)*dx)))
        info('Constraint: Sigma Zero: {:.2E}'.format(sigmazero))
    
        # Update internal data
        data.setdefault('Time', []).append(t)

        data.setdefault('Sigma Zero', []).append(sigmazero)

        data.setdefault('Norm of E', []).append(nor_E)
        data.setdefault('Absolute Error for E', []).append(err_E)
        data.setdefault('Relative Error for E (%)', []).append(helper.relative_error(err_E, nor_E))
        data.setdefault('Divergence of E', []).append(divfree_E)
        data.setdefault('Skew of E', []).append(skwfree_E)
        data.setdefault('Trace of E', []).append(trfree_E)
    
        data.setdefault('Norm of B', []).append(nor_B)
        data.setdefault('Absolute Error for B', []).append(err_B)
        data.setdefault('Relative Error for B (%)', []).append(helper.relative_error(err_B, nor_B))
        data.setdefault('Divergence of B', []).append(divfree_B)
        data.setdefault('Skew of B', []).append(skwfree_B)
        data.setdefault('Trace of B', []).append(trfree_B)

        # Evaluate functions at some point 
        # import numpy as np
        # pt_E = Eh(self.point)
        # nor_pt_E = np.sqrt(np.sum(np.square(pt_E)))
        # data.setdefault('Norm of E at point', []).append(nor_pt_E)
        # pt_B = Bh(self.point)
        # nor_pt_B = np.sqrt(np.sum(np.square(pt_B)))
        # data.setdefault('Norm of B at point', []).append(nor_pt_B)

        # Return current state
        current_data = OrderedDict()

        current_data['Sigma Zero'] = sigmazero

        current_data['E'] = (err_E, nor_E)
        # current_data['Absolute Error for E'] = err_E
        # current_data['Relative Error for E (%)'] = helper.relative_error(err_E, nor_E)
        current_data['Divergence of E'] = divfree_E
        current_data['Skew of E'] = skwfree_E
        current_data['Trace of E'] = trfree_E
    
        current_data['B'] = (err_B, nor_B)
        # current_data['Absolute Error for B'] = err_B
        # current_data['Relative Error for B (%)'] = helper.relative_error(err_B, nor_B)
        current_data['Divergence of B'] = divfree_B
        current_data['Skew of B'] = skwfree_B
        current_data['Trace of B'] = trfree_B

        return current_data

    def plot(self):
        mesh = self.W1.mesh()
        num_cells = mesh.size_global(mesh.geometry().dim())

        keys = self.data.keys()
        keys.remove('Time')
        for k in keys:
            helper.make_plot(self.data['Time'], self.data[k], ylabel = k, filename = str(num_cells) + "_" + k)

class Movie():
    """
    Save evolution of E and B to file.
    """

    def __init__(self, CG):
        self.solver_E = self.solver_B = None

        self.CG = CG
        mesh = CG.mesh()
        num_cells = mesh.size_global(mesh.geometry().dim())
        self.vtkfile_E = File("vtk_" + str(num_cells) + "/wave" + "_E" + ".pvd")
        self.vtkfile_B = File("vtk_" + str(num_cells) + "/wave" + "_B" + ".pvd")

    def save(self, ucur):

        (vtkE, vtkB) = ucur.split()
        (self.vtkfile_E, self.solver_E) << projection(vtkE, self.CG, self.solver_E)
        (self.vtkfile_B, self.solver_B) << projection(vtkB, self.CG, self.solver_B)

def compute(mesh, k):
    """
    Solve with given mesh and time step size.

    Input:  mesh    -- mesh
            k       -- time step size

    Output: err     -- L^2 error at final time
            nor     -- L^2 norm of exact solution at final time
    """

    deg = order_space

    if not periodicbc:
        constrained_domain = None
    else:
        # from helper_module import PeriodicBoundary3D as PB
        # constrained_domain = PB()
        from periodic_bc_examples import *
        constrained_domain = FlatTorus3D(1., 1., 1.)

    # Define function spaces for formulation
    # Space for weak divergence of E
    V0 = VectorFunctionSpace(mesh, 'CG', deg, constrained_domain = constrained_domain)
    # Spaces for E and B
    V1 = VectorFunctionSpace(mesh, 'N1curl', deg, constrained_domain = constrained_domain)
    V2 = VectorFunctionSpace(mesh, 'RT', deg, constrained_domain = constrained_domain)
    # Space for divergence of B
    V3 = VectorFunctionSpace(mesh, 'DG', deg-1, constrained_domain = constrained_domain)

    # Space to compute errors
    # W1 = TensorFunctionSpace(mesh, 'DG', deg+1, constrained_domain = constrained_domain)
    # W2 = TensorFunctionSpace(mesh, 'DG', deg+1, constrained_domain = constrained_domain)
    # W1 = VectorFunctionSpace(mesh, 'N1curl', deg+1, constrained_domain = constrained_domain)
    # W2 = VectorFunctionSpace(mesh, 'RT', deg+1, constrained_domain = constrained_domain)
    # W1 = VectorFunctionSpace(mesh, 'N1curl', deg, constrained_domain = constrained_domain)
    # W2 = VectorFunctionSpace(mesh, 'RT', deg, constrained_domain = constrained_domain)
    W1 = V1
    W2 = V2

    # Space to plot
    CG = TensorFunctionSpace(mesh, 'CG', 1, constrained_domain = constrained_domain)

    V = MixedFunctionSpace([V0, V1, V2])
    (sigma,E,B) = TrialFunctions(V)
    (tau,C,D) = TestFunctions(V)
    
    # Mass matrix
    m0 = inner(sigma,tau) * dx
    M0 = assemble(m0)
    # info("Assembling first part of mass matrix")
    m1 = inner(E,C) * dx
    M1 = assemble(m1)
    # info("Assembling second part of mass matrix")
    m2 = inner(B,D) * dx
    M2 = assemble(m2)
    M = M0 + M1 + M2
    
    # Bilinear form for CC. Assembling in parts is faster.
    a0 = ( inner(grad(sigma), C) - inner(E, grad(tau)) ) * dx
    A0 = assemble(a0)
    # info("Assembling first part of bilinear form")
    a1 = ( inner(curl(E[0,:]), D[0,:]) + inner(curl(E[1,:]), D[1,:]) + inner(curl(E[2,:]), D[2,:]) ) * dx
    A1 = assemble(a1)
    # info("Assembling second part of bilinear form")
    a2 = ( inner(B[0,:], curl(C[0,:])) + inner(B[1,:], curl(C[1,:])) + inner(B[2,:], curl(C[2,:])) ) * dx
    A2 = assemble(a2)
    A = A0 + A1 - A2
    L = M - .5*k*A
    R = M + .5*k*A

    # For initial conditions
    # Q = M + A
    
    # Define boundary to be entire boundary (for Dirichlet boundary conditions)
    def boundary(x, on_boundary):
        return on_boundary
    
    # Enforce BCs
    info("Creating BCs")
    if not periodicbc:
        bcs = [ DirichletBC(V.sub(0), j, boundary), DirichletBC(V.sub(1), g, boundary), DirichletBC(V.sub(2), h, boundary) ]

    # Solver for Q
    # if not periodicbc:
    #     for bc in bcs: bc.apply(L)
    # Qsolver = helper.get_solver(Q)

    # Solver for L
    if not periodicbc:
        for bc in bcs: bc.apply(L)
    Lsolver = helper.get_solver(L)

    # Solver for mass matrix
    if not periodicbc:
        for bc in bcs: bc.apply(M)
    Msolver = helper.get_mass_solver(M)
    # Msolver = helper.get_solver(M)
    
    # Initial data is L^2-projection
    info('Time step 0')
    tstep = 0
    t = f.t = j.t = g.t = h.t = t0
    uold = Function(V, name='u')
    F = assemble((inner(E0,C) + inner(B0,D)) * dx)
    if not periodicbc:
        for bc in bcs: bc.apply(F)
    Msolver.solve(uold.vector(), F)

    # # Initial conditions
    # info('Time step 0')
    # tstep = 0
    # t = f.t = g.t = h.t = t0
    # # info("Assembling mass matrix")
    # m0 = (inner(E0,C) + inner(B0,D)) * dx
    # M0 = assemble(m0)
    # # info("Assembling first part of bilinear form")
    # f1 = ( inner(curl(E0[0,:]), D[0,:]) + inner(curl(E0[1,:]), D[1,:]) + inner(curl(E0[2,:]), D[2,:]) ) * dx
    # F1 = assemble(f1)
    # # info("Assembling second part of bilinear form")
    # f2 = ( inner(B0[0,:], curl(C0[0,:])) + inner(B0[1,:], curl(C[1,:])) + inner(B0[2,:], curl(C[2,:])) ) * dx
    # F2 = assemble(f2)
    # uold = Function(V, name='u')
    # Qsolver.solve(uold.vector(), F)
    
    # Update source term
    foldvec = assemble(inner(f,C)*dx)

    info('Time {:.6F}. Step {:8d}.'.format(t, tstep))
    data_list = DataList(V0, W1, W2, V3)
    movie = Movie(CG)
    if saveframes: 
        movie.save(uold)
    if compute_errors: 
        data_list.update(t, uold, uex)
    
    # Compute solution
    info('Beginning time stepping. Final time {}.'.format(T))
    # Initial guess
    ucur = Function(V, name='u')
    ucur.assign(uold)
    while t + k <= T + helper.MY_EPS:
        # Crank-Nicolson time stepping:
        # (u_{n+1} - u_n)/k = (1/2) ( CC u_{n+1} + f_{n+1} + CC u_n + f_n )

        t += k
        tstep += 1

        timer = Timer('Time stepping')
    
        # Update source term and BCs
        f.t = j.t = g.t = h.t = t
        fcurvec = assemble(inner(f,C)*dx)
        F = R * uold.vector() + .5 * k * (fcurvec + foldvec)
        if not periodicbc:
            for bc in bcs: bc.apply(F)
    
        Lsolver.solve(ucur.vector(), F)
        timer.stop()
    
        uold.assign(ucur)
        foldvec = fcurvec
    
        # Do some tests a few times per unit time
        if not tstep % max(int(floor(1/k/30)), 10):
            info('Time {:.6F}. Step {:8d}.'.format(t, tstep))
            if saveframes: 
                movie.save(ucur)
            if compute_errors:
                data_list.update(t, ucur, uex)
            log(PROGRESS, 'Done computing errors and constraints')
    
    # if not saveframes: 
    #     movie.save(ucur)

    # Print error at final time
    info('Time {:.6F}. Step {:8d}.'.format(t, tstep))
    if compute_errors:
        data_list_final = data_list.update(t, ucur, uex)
        data_list.plot()
    elif compute_errors_at_end:
        data_list_final = data_list.update(t, ucur, uex)
    else:
        data_list_final = {}

    # Ouptut errors and number of steps
    return (data_list_final, tstep)

if __name__ == "__main__":
    """
    Create mesh, compute, refine.
    """

    nx = ny = nz = 4
    k = 1./120.
    # k = 1./80.
    mesh = UnitCubeMesh(nx, ny, nz)
    helper.refine_loop(mesh, k, compute, order_space = order_space, order_time = order_time, max_num_mesh = 7, max_cells = 10**7)
