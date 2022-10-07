'''
Poisson--Nernst--Planck/Navier--Stokes equations

Strong primal form (m,n are the concentrations of positively and negatively charged species, respectively) 

1/Sc*(d/dt u + grad(u)*u) - Delta(u) + grad(p) = -kappa/(2*eps**2)*(m-n)*grad(phi)
                                        div(u) = 0
 d/dt m + u.grad(m) - div(grad(m)+m*grad(phi)) = 0
 d/dt n + u.grad(n) - div(grad(n)-n*grad(phi)) = 0
                      -div(2*eps**2*grad(phi)) = m-n

BCs : 

On y=0 (membrane surface): 
    m=2, phi=0, [grad(n)-n*grad(phi)-n*u].nu = 0, u = 0

On y=1 (reservoir): 
    m = n = 1, phi=120, u = 0

On left, right: 
    periodic for u,phi,m,n

As velocity is prescribed everywhere, we fix the mean value of pressure with a real Lagrange multiplier

Kim et al use SUPG. Some form of stabilisation may be needed for higher Sc 

Formulation and parameters from Karatay et al and Wang et al (a bit clearer than Kim et al)

'''

from fenics import *
import numpy as np
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

fileO = XDMFFile("outputs/out-CoupledNS-PNP.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["flush_output"] = True

# ************ Time units *********************** #

time = 0.0; Tfinal = 1.e-2;  dt = 1.e-6; 
inc = 0;   frequency = 50;

# ****** Model coefficients ****** #

Sc  = Constant(1.e3) # Schmidt number
eps = Constant(2.e-3) # electrostatic screening length (no convergence with 1e-3)
kappa = Constant(0.5) # electrohydrodynamic coupling constant

phi0 = Constant(0.) # applied voltage bottom
phi1 = Constant(120.) # applied voltage top
m0   = Constant(2.) # boundary cation concentration bottom
mn1  = Constant(1.) # boundary concentration top 

# ********** Mesh construction and boundary labels **************** #

def gradedMesh(lx,ly, Nx, Ny):
    m = UnitSquareMesh(Nx, Ny,'crossed')
    x = m.coordinates()

    #Refine on bottom only
    x[:,1] = x[:,1] - 1
    x[:,1] = np.cos(np.pi * (x[:,1] - 1.) / 2.) + 1.      

    #Scale
    x[:,0] = x[:,0]*lx; x[:,1] = x[:,1]*ly

    return m

mesh = gradedMesh(2,1,120,90)

bdry = MeshFunction("size_t", mesh, 1)
bdry.set_all(0)
bot = 31; top =32; 
GTop   = CompiledSubDomain("near(x[1],1) && on_boundary")
GBot   = CompiledSubDomain("near(x[1],0) && on_boundary")
GTop.mark(bdry,top); GBot.mark(bdry,bot); 
ds = Measure("ds", subdomain_data = bdry)

class PeriodicBoundary(SubDomain):

    # Left boundary is target
    def inside(self, x, on_boundary):
        return bool(near(x[0],0) and on_boundary)

    # Map right boundary to target
    def map(self, x, y):
        y[0] = x[0] - 2.0
        y[1] = x[1]

# *********** Finite element spaces (MINI + Lagrange) ************* #

P1  = FiniteElement('CG', mesh.ufl_cell(), 1)
Bub = FiniteElement("Bubble", mesh.ufl_cell(), 3)
P1b = VectorElement(P1 + Bub)
R0  = FiniteElement('R', mesh.ufl_cell(), 0)

Hh  = FunctionSpace(mesh, MixedElement([P1b,P1,P1,P1,P1,R0]),\
                    constrained_domain=PeriodicBoundary())

print (" ****** Total DoF = ", Hh.dim())
    
# *********** Trial and test functions ********** #

Utrial = TrialFunction(Hh)
Usol = Function(Hh)
u, p, m, n, phi, xi = split(Usol)
v, q, r, s, psi, ze = TestFunctions(Hh)

#******** boundary conditions on top and bottom *********

# need to 'project' when dealing with enriched spaces
zv = project(Constant((0,0)),Hh.sub(0).collapse())

bcU1   = DirichletBC(Hh.sub(0), zv,   bdry, bot)
bcU2   = DirichletBC(Hh.sub(0), zv,   bdry, top)
bcM1   = DirichletBC(Hh.sub(2), m0,   bdry, bot)
bcM2   = DirichletBC(Hh.sub(2), mn1,  bdry, top)
bcN    = DirichletBC(Hh.sub(3), mn1,  bdry, top)
bcPhi1 = DirichletBC(Hh.sub(4), phi0, bdry, bot)
bcPhi2 = DirichletBC(Hh.sub(4), phi1, bdry, top)
bcs    = [bcU1,bcU2,bcM1,bcM2,bcN,bcPhi1,bcPhi2]

# ***** 2% of random perturbation on the initial conditions ***** # 

pert = Function(Hh.sub(3).collapse())
pert.vector()[:] = np.random.uniform(0.98,1.0,pert.vector().get_local().size)
mi0  = Expression("i*(2-x[1])", i=pert, degree=1)
ni0  = Expression("i*x[1]", i=pert,  degree=1)
mold = interpolate(mi0,Hh.sub(2).collapse())
nold = interpolate(ni0,Hh.sub(3).collapse())
uold = project(zv,Hh.sub(0).collapse())

# *************** Variational forms ***************** #

# momentum and mass 
FF = 1./(Sc*dt)*dot(u-uold,v)*dx \
    + 1./Sc*dot(grad(u)*u,v)*dx \
    + inner(grad(u),grad(v))*dx \
    - p * div(v) * dx \
    - q * div(u) * dx \
    + kappa/(2*eps**2)*(m-n)*dot(grad(phi),v)*dx
     
# fixing zero-average pressure with Lagrange multiplier
FF += p * ze * dx + q * xi * dx
 
# positive charge
FF += 1./dt*(m-mold)*r*dx \
    + dot(u,grad(m))*r*dx \
    + dot(grad(m)+m*grad(phi),grad(r))*dx
    
# negative charge
FF += 1./dt*(n-nold)*s*dx \
    + dot(u,grad(n))*s*dx \
    + dot(grad(n)-n*grad(phi),grad(s))*dx

# Poisson
FF += 2*eps**2*dot(grad(phi),grad(psi))*dx \
    -(m-n)*psi*dx

# Construction of tangent problem 
Tang = derivative(FF, Usol, Utrial)
problem = NonlinearVariationalProblem(FF, Usol, bcs, J=Tang)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'
solver.parameters['newton_solver']['linear_solver']      = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['maximum_iterations'] = 35

# ********* Time loop ************* #

while (time <= Tfinal):
    
    print("time = %1.3e" % time)
    solver.solve()
    uh, ph, mh, nh, phih, xih = Usol.split()
    assign(uold,uh); assign(mold,mh); assign(nold,nh); 

    if (inc % frequency == 0):
        uh.rename("u","u"); fileO.write(uh,time)
        ph.rename("p","p"); fileO.write(ph,time)
        mh.rename("m","m"); fileO.write(mh,time)
        nh.rename("n","n"); fileO.write(nh,time)
        phih.rename("phi","phi"); fileO.write(phih,time)
        
    inc += 1; time += dt
