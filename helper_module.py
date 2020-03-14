# Install tabulate with pip install tabulate --user

from __future__ import division
from tabulate import tabulate
from collections import OrderedDict

from dolfin import *

# Import matplotlib and pyplot
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
# Bigger font size
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt

# Tolerance for real numbers
MY_EPS = 10**(-5)
# LaTeX output
texfile_name = 'table.tex'

# Optimization
# parameters["form_compiler"]["optimize"] = True

# Print only from root
parameters["std_out_all_processes"] = False;

# Get MPI communicator
try:
    comm = mpi_comm_world()
    rank = MPI.rank(comm)
except:
    rank = 0

# Set log level
set_log_level(ERROR)
if not rank:
    set_log_level(PROGRESS)

class PeriodicBoundary3D(SubDomain):
    """
    Periodic boundary for cube.

    Use
        from periodic_bc_examples import *
        FlatTorus3D(1., 1., 1.)
    """
        
    def __init__(self, length = 1., length_scaling = 1.):
        SubDomain.__init__(self)
        # self.tol = tolerance = DOLFIN_EPS
        # self.length = length
        # self.length_scaling = length_scaling
        self.L = length/length_scaling

    def inside(self, x, on_boundary):
        # Return True if on front, left, bottom boundary AND NOT on one of the corners (L, 0, 0), (0, L, 0), (0, 0, L)
        L = self.L
        return ( near(x[0], 0.) or near(x[1], 0.) or near(x[2], 0.) and
                not (   (near(x[0], L) and near(x[1], 0.) and near(x[2], 0.)) or
                        (near(x[0], 0.) and near(x[1], L) and near(x[2], 0.)) or
                        (near(x[0], 0.) and near(x[1], 0.) and near(x[2], L)))
                and on_boundary )

    def map(self, x, y):
        # Map boundaries
        L = self.L
        if near(x[0], L) and near(x[1], L) and near(x[2], L):
            y[0] = x[0] - L
            y[1] = x[1] - L
            y[2] = x[2] - L
        elif near(x[0], 0.) and near(x[1], L) and near(x[2], L):
            y[0] = x[0]
            y[1] = x[1] - L
            y[2] = x[2] - L
        elif near(x[0], L) and near(x[1], 0.) and near(x[2], L):
            y[0] = x[0] - L
            y[1] = x[1]
            y[2] = x[2] - L
        elif near(x[0], L) and near(x[1], L) and near(x[2], 0.):
            y[0] = x[0] - L
            y[1] = x[1] - L
            y[2] = x[2]
        elif near(x[0], L):
            y[0] = x[0] - L
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], L):
            y[0] = x[0]
            y[1] = x[1] - L
            y[2] = x[2]
        elif near(x[2], L):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - L
        else:
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2]

class PeriodicBoundary2D(SubDomain):
    """
    Periodic boundary for square.
    http://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition
    """
        
    def __init__(self, length = 1., length_scaling = 1.):
        SubDomain.__init__(self)
        # self.tol = tolerance = DOLFIN_EPS
        # self.length = length
        # self.length_scaling = length_scaling
        self.L = length/length_scaling

    def inside(self, x, on_boundary):
        # Return True if on left, bottom boundary AND NOT on one of the corners (L, 0), (0, L)
        L = self.L
        return ( near(x[0], 0.) or near(x[1], 0.) and
                not (   (near(x[0], L) and near(x[1], 0.)) or
                        (near(x[0], 0.) and near(x[1], L)) )
                and on_boundary )

    def map(self, x, y):
        # Map boundaries
        L = self.L
        if near(x[0], L) and near(x[1], L):
            y[0] = x[0] - L
            y[1] = x[1] - L
        elif near(x[0], L):
            y[0] = x[0] - L
            y[1] = x[1]
        elif near(x[1], L):
            y[0] = x[0]
            y[1] = x[1] - L
        else:
            y[0] = x[0]
            y[1] = x[1]

def vskw(mat):
        """
        Extract the skew part of a matrix. 

        Input: FEniCS matrix
        Output: skew part as scalar/vector
        """

        skw = skew(mat)
        if mat.geometric_dimension() == 2:
            return skw[0,1]
        elif mat.geometric_dimension() == 3:
            return as_vector((skw[1,2], skw[2,0], skw[0,1]))
        else:
             error('Incorrect mesh dimension.')

def compress_matrix(A):
    """
    Compress FEniCS-PETSc matrix A to reduce pseudo-nonzeros.
    """

    C = PETScMatrix()
    A.compressed(C)

    return C

def get_memory_usage():
    """
    Return maximal memory usage in MB since the beginning of the program.

    Input:  --
    Output: mem2    -- maximal memory usage in MB
    """
    
    conversion_factor = float(1024)          # for MB
    # conversion_factor = float(1024*1024)   # for GB

    # FEniCS memory usage command
    (_, mem2) = memory_usage(as_string = False)
    mem2 = mem2 / conversion_factor
    
    return mem2

def get_solver(M):
    """
    Define iterative solver for given operator M.

    Input:  M       -- operator to invert
    Ouput:  solver  -- solver for given operator
    """
    
    # Compute LU factorization of M
    # solver = PETScLUSolver('mumps')
    # solver = PETScLUSolver()

    # solver.parameters["reuse_factorization"] = True

    solver = PETScKrylovSolver('minres', 'amg')

    # solver.parameters['preconditioner']['structure'] = 'same'
    # solver.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
    # Use last time step as initial guess
    solver.parameters['nonzero_initial_guess'] = True

    # solver.parameters['absolute_tolerance'] = 1E-07 # 1.E-7 ? 1.E-15
    # solver.parameters['relative_tolerance'] = 1E-06 # 1.E-6 ? 1.E-06
    # solver.parameters['divergence_limit'] = 1.E6
    # solver.parameters['maximum_iterations'] = 20000

    # solver.parameters['error_on_nonconvergence'] = True

    solver.set_operator(M)

    return solver

def get_mass_solver(M):
    """
    Define iterative solver for given operator M.

    Input:  M       -- operator to invert
    Ouput:  solver  -- solver for given operator
    """
    
    # Compute LU factorization
    # solver = PETScLUSolver()
    # solver = PETScLUSolver('mumps')

    # solver.parameters["reuse_factorization"] = True

    # solver = PETScKrylovSolver('cg', 'jacobi')
    solver = PETScKrylovSolver('default', 'default')

    # solver = PETScKrylovSolver('minres', 'default')
    # solver = PETScKrylovSolver('minres', 'amg')
    # solver = PETScKrylovSolver('gmres', 'amg')
    # solver = PETScKrylovSolver('cg', 'default')

    # solver.parameters['absolute_tolerance'] = 1E-07 # 1.E-7 ? 1.E-15
    # solver.parameters['relative_tolerance'] = 1E-06 # 1.E-6 ? 1.E-06
    # solver.parameters['divergence_limit'] = 1.E6
    # solver.parameters['maximum_iterations'] = 20000

    # Use last time step as initial guess
    solver.parameters['nonzero_initial_guess'] = True

    solver.parameters['error_on_nonconvergence'] = True
    # solver.parameters['preconditioner']['structure'] = 'same'
    # solver.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'

    solver.set_operator(M)

    return solver

def projection(f, V, solver = None):
    """
    L^2 projection for f in V using mass matrix solver.
    
    Input:  f -- function to project
            V -- space in which to project
            solver -- linear solver used for projection
    Output  f -- projected function
            solver -- linear solver used for projection
    """

    u = TrialFunction(V)
    v = TestFunction(V)

    if solver is None:
        M = assemble(inner(u, v)*dx)
        solver = get_mass_solver(M)

    F = assemble(inner(f, v)*dx)
    f = Function(V, name='u')
    solver.solve(f.vector(), F)

    return (f, solver)

def hcurl_zero_hodge_decomp_2(f, V0, V1):
    """
    Return the curl part of the Hodge decomposition for H0(curl).

    Thanks to Larry, 2015-04-21.

    Input:
        f   -- vector function
        V0  -- CG
        V1  -- N1curl
    Output:
        h   -- the curl part 

        # (f, h, g) -  f interpolated 
        #              h the curl part
        #              g the grad part
    """

    f = interpolate(f, V1)

    # Set system
    u = TrialFunction(V0)
    v = TestFunction(V0)
    a = inner(grad(u), grad(v)) * dx
    A = assemble(a)
    b = assemble(inner(f, grad(v))*dx)

    # Use entire boundary for Dirichlet BC
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V0, Constant('0.'), boundary)
    bc.apply(A, b)

    # Solve
    u = Function(V0)
    solver = PETScKrylovSolver("minres", "amg")
    solver.solve(A, u.vector(), b)

    # Remove gradient part
    g = project(grad(u), V1)
    h = Function(V1)
    h.vector()[:] = f.vector() - g.vector()

    return h

# def hcurl_zero_hodge_decomp(f, V0, V1):
def hcurl_zero_hodge_decomp(f, mesh, deg):
    """
    Return the curl part of the Hodge decomposition for H0(curl).

    Thanks to Larry, 2015-04-21.

    Input:
        f   -- vector function
        V0  -- CG
        V1  -- N1curl
    Output:
        h   -- the curl part 

        # (f, h, g) -  f interpolated 
        #              h the curl part
        #              g the grad part
    """

    V0 = FunctionSpace(mesh, 'CG', deg)
    V1 = FunctionSpace(mesh, 'N1curl', deg)

    # Interpolate f
    f = interpolate(f, V1)

    # Define Hodge Laplacian
    u = TrialFunction(V0)
    v = TestFunction(V0)
    a = (inner(sigma, tau) + inner(u, grad(tau)) + inner(grad(sigma), v) + inner(curl(u), curl(v)) ) * dx
    A = assemble(a)
    b = assemble(inner(curl(f), v)*dx)

    # Use entire boundary for Dirichlet BC
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V0, Constant('0.'), boundary)
    bc.apply(A, b)

    # Solve
    u = Function(V0)
    solver = PETScKrylovSolver("minres", "amg")
    solver.solve(A, u.vector(), b)

    # Remove the gradient part
    g = project(grad(u), V1)
    h = Function(V1)
    h.vector()[:] = f.vector() - g.vector()

    # return (f, h, g)
    return h

def hdiv_zero_hodge_decomp(f, mesh, deg):
    """
    Return the curl part of the Hodge decomposition for H0(curl).

    Thanks to Larry, 2015-04-21.

    Input:
        f   -- vector function
        V0  -- CG
        V1  -- N1curl
    Output:
        h   -- the curl part 

        # (f, h, g) -  f interpolated 
        #              h the curl part
        #              g the grad part
    """

    V1 = FunctionSpace(mesh, 'N1curl', deg)
    V2 = FunctionSpace(mesh, 'RT', deg)

    # Interpolate f
    f = interpolate(f, V2)

    # Define Hodge Laplacian
    u = TrialFunction(V1)
    v = TestFunction(V1)
    a = dot(curl(u), curl(v)) * dx
    A = assemble(a)
    b = assemble(dot(f, curl(v))*dx)

    # Use entire boundary for Dirichlet BC
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V1, Constant(('0.', '0.', '0.')), boundary)
    bc.apply(A, b)

    # Solve
    u = Function(V1)
    solver = PETScKrylovSolver("minres", "amg")
    solver.solve(A, u.vector(), b)

    # Remove the gradient part
    g = project(curl(u), V2)
    h = Function(V2)
    h.vector()[:] = f.vector() - g.vector()

    # return (f, h, g)
    return h

def relative_error(error, norm):
    """
    Compute relative error. Return NaN if failing the calculation.

    Input:  error  -- error
            norm   -- norm
    Output: rel    -- relative error
    """

    from math import log as ln # (log is a dolfin name -- so is logg!)

    try:
        # Compute rate of convergence
        rel = 100*error/norm
    except (ZeroDivisionError):
        # If fails to compute rate r, assign NaN.
        rel = float('NaN')
    
    return rel

def monitor_error(uh, uex, W, solver = None):
    """
    Print error at final time.

    Input:  uh  -- approximate solution
            uex -- exact solution
            W   -- space to project uh and uex in
            solver  -- linear solver with operator already assembled

    Output: l2error -- L^2 error at current time
            l2norm  -- L^2 norm of exact solution at current time
            solver  -- linear solver used
    """

    timer = Timer('Computing error')

    u = TrialFunction(W)
    v = TestFunction(W)

    if solver is None:
        M = assemble(inner(u,v)*dx)
        solver = get_mass_solver(M)

    uapprox = Function(W)
    F = assemble(inner(uh,v)*dx)
    solver.solve(uapprox.vector(), F) 

    utrue = Function(W)
    F = assemble(inner(uex,v)*dx)
    solver.solve(utrue.vector(), F) 

    # uapprox = interpolate(uh, W)
    # utrue = interpolate(uex, W)

    l2error = sqrt(abs(assemble(inner(uapprox-utrue, uapprox-utrue)*dx)))
    l2norm = sqrt(abs(assemble(inner(utrue, utrue)*dx)))
    l2normapprox = sqrt(abs(assemble(inner(uapprox, uapprox)*dx)))
    info('L2 norm of exact {:.6E}, L2 norm of approximate {:.6E},\nL2 error {:.6E}, relative L2 error {:.6F}%'.format(
            l2norm, l2normapprox, l2error, relative_error(l2error, l2norm)))

    return (l2error, l2norm, solver)

def print_error(table):
    """
    Print a table of values tracked during evolution.

    Input:  table -- dictionary of values to track during evolution
    """

    # Display table
    floatfmt=".2E"
    tab = tabulate(table, headers="keys", tablefmt="rst", floatfmt=floatfmt)
    if not rank: 
        print(tab)

    # Write LaTeX table to file
    caption = 'Values tracked during evolution'
    width = '.75'
    if not rank:
        with open(texfile_name, "a") as texfile:
            texfile.write('\\bigskip\n')
            texfile.write('\\begin{table}\\centering\n\\begin{subtable}{' + width + '\\textwidth}\\centering\n')
            tab = tabulate(table, headers="keys", tablefmt="latex_booktabs", floatfmt=floatfmt)
            tab = tab.replace('\\toprule', '')
            texfile.write(tab)
            texfile.write('\n\\caption{' + caption + '}\n\\end{subtable}\n\\end{table}\n\n')

def make_plot(xcomp, ycomp, xlabel="Time", ylabel="Y", filename="generic"):
    """
    Create file containing generated plot.

    Input:  xcomp       -- x values
            ycomp       -- y values
            xlabel      -- label for x axis
            ylabel      -- label for y axis
            filename    -- name of file to containt the plot
    Output: --
    """

    filename = 'plot_' + filename + '.pdf'
    filename = filename.replace(' (%)', '')
    filename = filename.replace(' ', '_')

    # Create an empty figure with no axes
    fig = plt.figure()
    fig.clf()

    # Add plot
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    # Scatter plot
    ax.plot(xcomp, ycomp)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Find maximum of data
    xcompmax = xcomp[:].sort()
    xcompmax = xcomp[-1]
    ycompmax = ycomp[:].sort()
    ycompmax = ycomp[-1]

    # Set limit of x and y
    ax.set_xlim([0, xcompmax])
    ax.set_ylim(bottom = 0)

    # Save to file
    if not rank:
        fig.savefig(filename, bbox_inches='tight')

    # Close figure
    plt.close(fig)

    # Build table
    table = OrderedDict()
    table[xlabel] = xcomp
    table[ylabel] = ycomp

    # Display table
    floatfmt = ".2E"
    tab = tabulate(table, headers="keys", tablefmt="rst", floatfmt=floatfmt)
    if not rank:
        print(tab)

    # Write LaTeX table to file
    caption = ylabel + ' versus ' + xlabel
    # width = '.35'
    width = ''
    if not rank:
        with open(texfile_name, "a") as texfile:
            texfile.write('\\bigskip\n')
            texfile.write('\\begin{table}\\centering\n\\begin{subtable}{' + width + '\\textwidth}\\centering\n')
            tab = tabulate(table, headers="keys", tablefmt="latex_booktabs", floatfmt=floatfmt)
            tab = tab.replace('\\toprule', '')
            texfile.write(tab)
            texfile.write('\n\\caption{' + caption + '}\n\\end{subtable}\n\\end{table}\n\n')

def convergence_rate(error0, error1, h0, h1):
    """
    Compute rate of convergence going from error0 for a mesh of size h0 to error1 for a mesh of size h1. 
    Return NaN if failing the calculation.

    Input:  error0  -- error on first mesh
            error1  -- error on second mesh
            h0      -- size of first mesh
            h1      -- size of second mesh
    Output: r       -- rate of convergence
    """

    from math import log as ln # (log is a dolfin name -- so is logg!)

    try:
        # Compute rate of convergence
        r = ln(error1/error0)/ln(h1/h0)
    except (ZeroDivisionError,ValueError):
        # If fails to compute rate r, assign NaN.
        r = float('NaN')
    
    return r

def print_errrate(h_list, errnorm_list, label = ''):
    """
    Print mesh size and error information.

    Input:  h_list      -- list of mesh sizes
            errnorm_list   -- list of (error, norm of exact solution)
    Output: --
    """

    istuple = True
    try:
        # Convert list of tuples into two lists
        error_list, norm_list = zip(*errnorm_list)
    except TypeError:
        error_list = errnorm_list
        istuple = False

    # Compute rate of convergence
    if len(error_list) > 1:
        r_list = [convergence_rate(error_list[i-1], error_list[i], h_list[i-1], h_list[i]) for i in range(1, len(error_list))]
        r_list.insert(0, float('NaN'))
    else:
        r_list = [ float('NaN') ]

    if istuple:
        # Compute relative error
        rel_list = [relative_error(error_list[i], norm_list[i]) for i in range(len(error_list))]

    # Format numbers
    org = '.'
    pad = '@'
    # h_list = ['{:.2E}'.format(i).replace(org,pad) for i in h_list]
    error_list = ['{:.2E}'.format(i).replace(org,pad) for i in error_list]
    if istuple:
        rel_list = ['{:.2F}%'.format(rel_list[i]).replace(org,pad) for i in range(len(error_list))]
        norm_list = ['{:.2E}'.format(i).replace(org,pad) for i in norm_list]
    r_list = ['{:+2.2F}'.format(i).replace(org,pad) for i in r_list]
    r_list[0] = ''

    # Build table
    table = OrderedDict()
    # table['Mesh'] = h_list 
    if istuple:
        # table['Relative Error (%) for ' + label] = rel_list
        table['Relative Error for ' + label] = rel_list
        table['Norm of ' + label] = norm_list
        table['Absolute Error for ' + label] = error_list
    else:
        table[label] = error_list
    table['Rate'] = r_list 

    # Display table
    tab = tabulate(table, headers="keys", tablefmt="rst")
    tab = tab.replace(pad,org)   # Remove padding
    if not rank:
        print(tab)

    # Write LaTeX table to file
    caption = 'Errors for ' + label
    # width = '.35'
    width = ''
    if not rank:
        with open(texfile_name, "a") as texfile:
            texfile.write('\\bigskip\n')
            texfile.write('\\begin{table}\\centering\n\\begin{subtable}{' + width + '\\textwidth}\\centering\n')
            tab = tabulate(table, headers="keys", tablefmt="latex_booktabs")
            tab = tab.replace(pad,org)   # Remove padding
            tab = tab.replace('\\toprule', '')
            texfile.write(tab)
            texfile.write('\n\\caption{' + caption + '}\n\\end{subtable}\n\\end{table}\n\n')

def print_infocode(table):
    """
    Print current table on the screen and in tex file.

    Input:  table -- ordered dictionary of equal lenght list
    Output: --
    """

    # Display table
    if not rank:
        print(tabulate(table, headers="keys", tablefmt="rst"))

    # Write LaTeX table to file
    caption = 'Code information about each run'
    # width = '.35'
    width = ''
    if not rank:
        with open(texfile_name, "a") as texfile:
            texfile.write('\\bigskip\n')
            texfile.write('\\begin{table}\\centering\n\\begin{subtable}{' + width + '\\textwidth}\\centering\n')
            tab = tabulate(table, headers="keys", tablefmt="latex_booktabs")
            tab = tab.replace('\\toprule', '')
            texfile.write(tab)
            texfile.write('\n\\caption{' + caption + '}\n\\end{subtable}\n\\end{table}\n\n')

def refine_loop(mesh, k, compute, order_space = 1, order_time = 1, max_num_mesh = 1, max_cells = 10**5):
    """
    Compute on multiple refined meshes starting with given mesh and time step.

    Input:  mesh            -- starting mesh to be refined
            k               -- starting time step to be refined
            compute         -- function mapping (mesh, k) to (error, norm)
            max_num_mesh    -- maximal number of mesh to compute on
            max_cells       -- maximal number of cells a mesh can have
    Output: --
    """

    info('Order in time: {}. Order in space: {}.'.format(order_time, order_space))
    time_refine = order_space/order_time

    infocode = OrderedDict()
    infocode['Mesh size'] = []      # List of element sizes
    infocode['Cell count'] = []  # List of number of cells
    infocode['Time step'] = []      # List of time steps
    infocode['Step count'] = []  # List of number of steps
    infocode['Timing'] = [] # List of time for computation for each mesh
    infocode['Memory'] = [] # List of memory usage for each mesh
    data_list = OrderedDict()   # List of errors

    num_mesh = 0
    while mesh.num_cells() < max_cells:
        """
        Compute solution on mesh, and refine until mesh has too many cells.
        """

        # Print current date and time 
        from datetime import datetime
        if not rank:
            print("\n" + datetime.now().isoformat())
    
        # Display mesh information
        mesh.init()
        info('\nMesh with {} cells, {} faces, {} edges, {} vertices.'.format(
            mesh.num_cells(), mesh.num_faces(), mesh.num_edges(), mesh.num_vertices()))
        # Record mesh size
        h = mesh.hmax() 
        infocode['Mesh size'].append(h)
        infocode['Time step'].append(k)
        num_cells = mesh.size_global(mesh.geometry().dim())
        infocode['Cell count'].append(num_cells)
        info('Refine iteration: {}. Mesh size: {}. Step size: {}.\n'.format(num_mesh, h, k))
    
        # Compute error on given mesh
        timer = Timer('Compute on mesh')
        (data, stepcount) = compute(mesh, k)
        infocode['Timing'].append(int(timer.stop()))

        # Record data for the mesh
        for key in data.keys():
            data_list.setdefault(key, []).append(data[key])
        infocode['Step count'].append(stepcount)
        # Record maximal memory usage
        infocode['Memory'].append(int(get_memory_usage()))
    
        # Output errors, norms, for each mesh size
        for key in data_list.keys():
            print_errrate(infocode['Mesh size'], data_list[key], label = key)
        # Output timings, memory usages, for each mesh size
        print_infocode(infocode)
        list_timings(TimingClear_keep, [TimingType_wall, TimingType_user, TimingType_system])
    
        # Refine mesh and time step
        num_mesh += 1
        if num_mesh >= max_num_mesh: break
        if mesh.num_cells() * 2**(mesh.geometry().dim()) > max_cells: break # Estimate number of cells after refining
        mesh = refine(mesh)
        k = k/(2**time_refine)
