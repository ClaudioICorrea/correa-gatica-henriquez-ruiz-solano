'''
Poisson--Nernst--Planck equations

Strong primal form (m,n are the concentrations of positively and negatively charged species, respectively) 

        d/dt m - div(grad(m)+m*grad(phi)) = 0
        d/dt n - div(grad(n)-n*grad(phi)) = 0
                      -div(2*e**2*grad(phi)) = m-n

BCs : 

On y=0 (membrane surface): 
    m = m0, phi = 0, [grad(n)-n*grad(phi)-n*u].nu = 0

On y=1 (reservoir): 
    m = n = 1, phi=120

On walls: 
    periodic 

'''
from fenics import *
import numpy as np
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

fileO = XDMFFile("outputs/out-Ex06OnlyPNP.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["flush_output"] = True

 
def mymesh(lx,ly, Nx, Ny):
    m = UnitSquareMesh(Nx, Ny,'crossed')
    x = m.coordinates()

    #Refine on bottom only
    x[:,1] = x[:,1] - 1
    x[:,1] = np.cos(np.pi * (x[:,1] - 1.) / 2.) + 1.      

    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly

    return m


# ************ Time units *********************** #

time = 0.0; Tfinal = 1.;  dt = 1.e-3; 
inc = 0;   frequency = 20;

# ****** Model coefficients ****** #

Sc  = Constant(1.e3) # Schmidt number
ep  = Constant(1.5e-2) # electrostatic screening length
kappa = Constant(0.5) # electrohydrodynamic coupling constant

phi0 = Constant(30.) # applied voltage bottom
phi1 = Constant(0.) # applied voltage top
m0   = Constant(2.) # boundary cation concentration bottom
mn1  = Constant(1.) # boundary concentration top 

Dm = Constant(1.0)#34)
Dn = Constant(1.0)#2.03)

# ********** Mesh construction **************** #

mesh = mymesh(2,1,80,60)

# ******* Boundaries ******** #

bdry = MeshFunction("size_t", mesh, 1)
bdry.set_all(0)
bot = 31; top =32; 
GTop   = CompiledSubDomain("(x[1]>1.-DOLFIN_EPS) && on_boundary")
GBot   = CompiledSubDomain("(x[1]<DOLFIN_EPS) && on_boundary")
GTop.mark(bdry,top); GBot.mark(bdry,bot); 
ds = Measure("ds", subdomain_data = bdry)

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 2.0
        y[1] = x[1]

pbc = PeriodicBoundary()

# *********** Finite Element spaces ************* #

P1  = FiniteElement('CG', mesh.ufl_cell(), 1)


Mh = FunctionSpace(mesh, MixedElement([P1,P1,P1]), constrained_domain=pbc)
Ph = FunctionSpace(mesh,P1,constrained_domain=pbc)

print (" ****** Total DoF = ", Mh.dim())
    
# *********** Trial and test functions ********** #


Mtrial = TrialFunction(Mh)
Msol = Function(Mh)
m_, n_, phi_ = split(Msol)
r_, s_, psi_ = TestFunctions(Mh)


#******** boundary conditio
 
bcM1_   = DirichletBC(Mh.sub(0), m0, bdry, bot)
bcM2_   = DirichletBC(Mh.sub(0), mn1, bdry, top)
bcN_    = DirichletBC(Mh.sub(1), mn1, bdry, top)
bcPhi1_ = DirichletBC(Mh.sub(2), phi0, bdry, bot)
bcPhi2_ = DirichletBC(Mh.sub(2), phi1, bdry, top)

bcs_ = [bcM1_,bcM2_,bcN_,bcPhi1_,bcPhi2_]

# *************** Variational forms ***************** #
min0 = Expression("2-x[1]",  degree=1)
nin0 = Expression("x[1]",  degree=1)
mold_ = interpolate(min0,Mh.sub(0).collapse())
nold_ = interpolate(nin0,Mh.sub(1).collapse())


FF = 1./dt*(m_-mold_)*r_*dx + 1./dt*(n_-nold_)*s_*dx \
    + Dm*dot(grad(m_)+m_*grad(phi_),grad(r_))*dx \
    + Dn*dot(grad(n_)-n_*grad(phi_),grad(s_))*dx \
    + 2*ep**2*dot(grad(phi_),grad(psi_))*dx \
    -(m_-n_)*psi_*dx 
    
    

Tang = derivative(FF, Msol, Mtrial)
problem = NonlinearVariationalProblem(FF, Msol, bcs_, J=Tang)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'
solver.parameters['newton_solver']['linear_solver']      = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['maximum_iterations'] = 35


solver.solve()
mh_, nh_, phih_ = Msol.split()
mh_.rename("m","m"); fileO.write(mh_,time)
nh_.rename("n","n"); fileO.write(nh_,time)
phih_.rename("phi","phi"); fileO.write(phih_,time)
