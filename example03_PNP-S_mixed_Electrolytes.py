'''
Poisson--Nernst--Planck/Stokes equations

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

fileO = XDMFFile("outputs/out-CoupledS-PNP-mixed.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["flush_output"] = True

# ************ Time units *********************** #

time = 0.0; Tfinal = 1.1e-3;  dt = 1.e-6; 
inc = 0;   frequency = 20;

# ****** Model coefficients ****** #
#sc 1e3 eps 2e-3
Sc  = Constant(5e2) # Schmidt number
eps = Constant(2.2e-3) #2.5e-3
kappa = Constant(0.5) # electrohydrodynamic coupling constant
chibot = Constant(0.) # applied voltage bottom
chitop = Constant(120.) # applied voltage top
xi1bot = Constant(2.) # boundary cation concentration bottom
xitop  = Constant(1.) # boundary concentration top 


#NEW

epshat = Constant(2*eps**2)
dim    = 2
I      = Identity(dim)

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
n = FacetNormal(mesh)

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

k = 1        
# *********** Finite element spaces (MINI + Lagrange) ************* #
H = FiniteElement('RT', mesh.ufl_cell(), k+1) 
Q = VectorElement('DG', mesh.ufl_cell(), k)
X = FiniteElement('RT', mesh.ufl_cell(), k+1)
M = FiniteElement('DG', mesh.ufl_cell(), k)
R_ = FiniteElement('R', mesh.ufl_cell(), 0)
Vh = FunctionSpace(mesh, MixedElement([H,H,Q,X,M,X,M,X,M,R_]),\
                    constrained_domain=PeriodicBoundary())

    
print (" ****** Total DoF = ", Vh.dim())
    
# *********** Trial and test functions ********** #
trial = TrialFunction(Vh)
sol   = Function(Vh)

#H  ,H   ,Q,  X,  M,   X,   M,   X,   M,R
sigx,sigy,u,phi,chi,sig1,xi1,sig2,xi2 ,zeta = split(sol)
taux,tauy,v,psi,lam,tau1,eta1,tau2,eta2,theta  = TestFunctions(Vh)

tau = as_tensor((taux,tauy))
sig = as_tensor((sigx,sigy))
#******** boundary conditions *********

# u: 0 on bot and top => No condition for u since it will be natural DONE
# xi1=2 on bot, 1 on top => naturally DONE 
# xi2: 1 on top, zero flux on bot => naturally on top + sig2 is zero on bot essentially DONE
# chi : chi0 on bot, chi1 on top => naturally  DONE

bcSig2 = DirichletBC(Vh.sub(7), Constant((0,0)), bdry, bot)
bcs    = [bcSig2]

# ***** 2% of random perturbation on the initial conditions ***** # 

pert = Function(Vh.sub(8).collapse())
pert.vector()[:] = np.random.uniform(0.98,1.0,pert.vector().get_local().size)
xi10  = Expression("i*(2-x[1])", i=pert, degree=1)
xi20  = Expression("i*x[1]", i=pert,  degree=1)
xi1old = interpolate(xi10,Vh.sub(6).collapse())
xi2old = interpolate(xi20,Vh.sub(8).collapse())
uold = Function(Vh.sub(2).collapse())

# ********* Variational form ********* #

# Stokes
FF = - 1./(Sc*dt)*dot(u-uold,v)*dx \
     + inner(dev(sig),dev(tau))*dx \
     + dot(u,div(tau))*dx \
     + dot(v,div(sig))*dx \
     - kappa/epshat*(xi1-xi2)*dot(phi/epshat,v)*dx

# fixing zero-average pressure with Lagrange multiplier
FF += tr(sig) * theta * dx \
     + tr(tau) * zeta * dx

#positive charge 
FF += - 1./dt*(xi1-xi1old)*eta1*dx \
      + eta1*div(sig1)*dx \
      + dot(sig1,tau1)*dx \
      + xi1*div(tau1)*dx \
      - dot(xi1/epshat*phi - xi1*u,tau1)*dx \
      - dot(tau1,n)*xi1bot*ds(bot) \
      - dot(tau1,n)*xitop*ds(top)

# negative charge
FF += - 1./dt*(xi2-xi2old)*eta2*dx \
      + eta2*div(sig2)*dx \
      + dot(sig2,tau2)*dx \
      + xi2*div(tau2)*dx \
      - dot(-xi2/epshat*phi - xi2*u,tau2)*dx \
      - dot(tau2,n)*xitop*ds(top)

# Poisson 
FF += lam*div(phi)*dx \
      + lam*(xi1-xi2)*dx \
      + 1./epshat*dot(phi,psi)*dx \
      + chi*div(psi)*dx \
      - dot(psi,n)*chibot*ds(bot) \
      - dot(psi,n)*chitop*ds(top) 
     

# *************** Variational forms ***************** #

Tang = derivative(FF, sol, trial)
problem = NonlinearVariationalProblem(FF, sol, bcs, J=Tang)
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
    sigx_h,sigy_h,u_h,phi_h,chi_h,sig1_h,xi1_h,sig2_h,xi2_h,zeta_h = sol.split()

    assign(uold,u_h); assign(xi1old,xi1_h); assign(xi2old,xi2_h); 

    sig_h = as_tensor((sigx_h,sigy_h))
    Ph = FunctionSpace(mesh, 'DG', k)
    p_h = project(-1./dim*tr(sig_h), Ph)

    if (inc % frequency == 0):
        u_h.rename("u","u"); fileO.write(u_h,time)
        p_h.rename("p","p"); fileO.write(p_h,time)
        xi1_h.rename("xi1","xi1"); fileO.write(xi1_h,time)
        xi2_h.rename("xi2","xi2"); fileO.write(xi2_h,time)
        phi_h.rename("phi","phi"); fileO.write(phi_h,time)
        chi_h.rename("chi","chi"); fileO.write(chi_h,time)
        
    inc += 1; time += dt
