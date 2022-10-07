'''
Convergence test for a mixed scheme of the Navier--Stokes--Brinkman equations
The domain is (0,1)x(0,1)

Manufactured smooth solutions
#######################################
strong primal form: 

 eta*u - lam*div(mu*eps(u)) + grad(u)*u + grad(p) = f  in Omega
                                           div(u) = 0  in Omega 

Pure Dirichlet conditions for u 
                                                u = u_D on Gamma

Lagrange multiplier to fix the average of p
                                           int(p) = 0

######################################

strong mixed form in terms of (t,sigma,u,gamma)

                 t + gamma = grad(u) 
lam*mu*t - dev(u otimes u) = dev(sigma)
        eta*u - div(sigma) = f
+ BC:
                         u = u_D on Gamma
+ trace of pressure:
 int(tr(sigma+u otimes u)) = 0

'''

from fenics import *
import sympy2fenics as sf
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

fileO = XDMFFile("outputs/convergence-AFW-2D.xdmf")
fileO.parameters['rewrite_function_mesh']=True
fileO.parameters["functions_share_mesh"] = True

# Constant coefficients 
ndim = 2
Id = Identity(ndim)

lam = Constant(0.2)


# Macro operators 

epsilon = lambda v: sym(grad(v))
skewgr  = lambda v: grad(v) - epsilon(v)
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))

# Manufactured solutions as strings 

u_str = '(cos(pi*x)*sin(pi*y),-sin(pi*x)*cos(pi*y))'
p_str = 'sin(x*y)'

# Initialising vectors for error history 

nkmax = 6; # max refinements
l = 0 # polynomial degree

hvec = []; nvec = []; erru = []; rateu = []; errp = []; ratep = [];
errt = []; ratet = []; errsigma = []; ratesigma = []
errgamma = []; rategamma = []; 

rateu.append(0.0); ratet.append(0.0); rategamma.append(0.0);
ratep.append(0.0); ratesigma.append(0.0);

# Error history 

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1) 
    mesh = UnitSquareMesh(nps,nps)
    nn   = FacetNormal(mesh)
        
    hvec.append(mesh.hmax())

    # Heterogeneous viscosity 

    mu = Expression('exp(-x[0]*x[1])', degree=3, domain = mesh)
    eta = Expression('2 +sin(x[0]*x[1])', degree = 3, domain = mesh)
    
    # Instantiation of exact solutions
    
    u_ex  = Expression(str2exp(u_str), degree=l+4, domain=mesh)
    p_ex  = Expression(str2exp(p_str), degree=l+4, domain=mesh)

    t_ex = epsilon(u_ex)
    gamma_ex = skewgr(u_ex)
    sigma_ex = lam*mu*t_ex - outer(u_ex,u_ex) - p_ex * Id

    f_ex = eta*u_ex - div(sigma_ex)
    
    # Finite element subspaces

    Ht = VectorElement('DG', mesh.ufl_cell(), l+1, dim = 3)
    Hsig = FiniteElement('BDM', mesh.ufl_cell(), l+1)# In FEniCS, Hdiv tensors need to be defined row-wise
    Hu = VectorElement('DG', mesh.ufl_cell(), l)
    Hgam = FiniteElement('DG', mesh.ufl_cell(), l)
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)

    #product space
    Hh = FunctionSpace(mesh, MixedElement([Ht,Hsig,Hsig,Hu,Hgam,R0]))
    nvec.append(Hh.dim())
    
    # Trial and test functions (nonlinear setting)
    Trial = TrialFunction(Hh)
    Sol   = Function(Hh) 
    t_, sig1, sig2,u,gam_,xi = split(Sol)
    s_, tau1, tau2,v,del_,zeta = TestFunctions(Hh)

    t = as_tensor(((t_[0], t_[1]),(t_[2],-t_[0])))
    s = as_tensor(((s_[0], s_[1]),(s_[2],-s_[0])))

    sigma = as_tensor((sig1,sig2))
    tau   = as_tensor((tau1,tau2))

    gamma = as_tensor(((0,gam_),(-gam_,0)))
    delta = as_tensor(((0,del_),(-del_,0)))
                      
    # Essential boundary conditions: NONE FOR THIS CASE

    # Variational forms

    a   = lam*mu*inner(t,s)*dx 
    b1  = - inner(sigma,s)*dx
    b   = - inner(outer(u,u),s)*dx
    b2  = inner(t,tau)*dx
    bbt = dot(u,div(tau))*dx + inner(gamma,tau)*dx
    bb  = dot(div(sigma),v)*dx + inner(sigma,delta)*dx
    cc  = eta * dot(u,v)*dx

    #+ xi*tr(tau+outer(v,v))*dx ???
    AA = a + b1 + b2 + b + bbt + bb - cc + zeta*tr(sigma+outer(u,u))*dx + xi*tr(tau)*dx 
    FF = dot(tau*nn,u_ex)*ds - dot(f_ex,v)*dx + zeta*tr(sigma_ex+outer(u_ex,u_ex))*dx
    
    
    Nonl = AA - FF
    # Solver specifications (including essential BCs if any)

    Tangent = derivative(Nonl, Sol, Trial)
    Problem = NonlinearVariationalProblem(Nonl, Sol, J=Tangent)
    Solver  = NonlinearVariationalSolver(Problem)
    Solver.parameters['nonlinear_solver']                    = 'newton'
    Solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    Solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    Solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    Solver.parameters['newton_solver']['maximum_iterations'] = 25

    # Assembling and solving
    #solve(Nonl == 0, Sol)

    Solver.solve()
    th_, sigh1, sigh2,uh,gamh_,xih = Sol.split()
    
    th = as_tensor(((th_[0], th_[1]),(th_[2],-th_[0])))
    sigmah = as_tensor((sigh1,sigh2))
    gammah = as_tensor(((0,gamh_),(-gamh_,0)))

    Ph = FunctionSpace(mesh, 'DG', l)
    Th = TensorFunctionSpace(mesh, 'DG', l)
    # Postprocessing (eq 2.7)
    ph = project(-1/ndim*tr(sigmah + outer(uh,uh)),Ph) 

    sig_v = project(sigmah, Th)
    t_v = project(th, Th)
    
    # saving to file

    #uh.rename("u","u"); fileO.write(uh,nk*1.0)
    #t_v.rename("t","t"); fileO.write(t_v,nk*1.0)
    #sig_v.rename("sig","sig"); fileO.write(sig_v,nk*1.0)
    #ph.rename("p","p"); fileO.write(ph,nk*1.0)
   
    # Error computation

    E_t = assemble(inner(t_ex - th,t_ex-th)*dx)
    E_sigma_0 = assemble(inner(sigma_ex-sigmah,sigma_ex-sigmah)*dx)
    E_sigma_div = assemble(dot(div(sigma_ex-sigmah),div(sigma_ex-sigmah))**(2./3.)*dx)
    E_u = assemble(dot(u_ex-uh,u_ex-uh)**2*dx)
    E_gamma = assemble(inner(gamma_ex-gammah,gamma_ex-gammah)*dx)
    E_p = assemble((p_ex - ph)**2*dx)

    errt.append(pow(E_t,0.5))
    errsigma.append(pow(E_sigma_0,0.5)+pow(E_sigma_div,0.75))
    erru.append(pow(E_u,0.25))
    errgamma.append(pow(E_gamma,0.5))
    errp.append(pow(E_p,0.5))

    # Computing convergence rates
    
    if(nk>0):
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rategamma.append(ln(errgamma[nk]/errgamma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
       
# Generating error history 
print('==============================================================================================================')
print('   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  ')
print('==============================================================================================================')

for nk in range(nkmax):
    print('{:6d}  {:.4f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f} '.format(nvec[nk], hvec[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk], erru[nk], rateu[nk], errgamma[nk], rategamma[nk], errp[nk], ratep[nk]))
print('==============================================================================================================')


'''

   305 & 0.5000 & 6.72e-01 & 0.00 & 2.61e+00 & 0.00 & 2.96e-01 & 0.00 & 8.28e-01 & 0.00 & 1.68e-01 & 0.00 
  1185 & 0.2500 & 3.10e-01 & 1.12 & 1.30e+00 & 1.00 & 1.61e-01 & 0.87 & 4.16e-01 & 0.99 & 7.24e-02 & 1.22 
  4673 & 0.1250 & 1.53e-01 & 1.02 & 6.45e-01 & 1.01 & 8.13e-02 & 0.99 & 2.09e-01 & 1.00 & 2.81e-02 & 1.36 
 18561 & 0.0625 & 7.65e-02 & 1.00 & 3.22e-01 & 1.00 & 4.07e-02 & 1.00 & 1.04e-01 & 1.00 & 1.27e-02 & 1.15 
 73985 & 0.0312 & 3.82e-02 & 1.00 & 1.61e-01 & 1.00 & 2.04e-02 & 1.00 & 5.22e-02 & 1.00 & 6.17e-03 & 1.04 
295425 & 0.0156 & 1.91e-02 & 1.00 & 8.05e-02 & 1.00 & 1.02e-02 & 1.00 & 2.61e-02 & 1.00 & 3.06e-03 & 1.01 

   697 & 0.5000 & 1.27e-01 & 0.00 & 5.59e-01 & 0.00 & 6.38e-02 & 0.00 & 1.60e-01 & 0.00 & 2.59e-02 & 0.00 
  2737 & 0.2500 & 3.24e-02 & 1.98 & 1.35e-01 & 2.05 & 1.77e-02 & 1.85 & 4.13e-02 & 1.96 & 7.39e-03 & 1.81 
 10849 & 0.1250 & 7.93e-03 & 2.03 & 3.35e-02 & 2.01 & 4.46e-03 & 1.99 & 1.04e-02 & 1.99 & 1.77e-03 & 2.06 
 43201 & 0.0625 & 1.95e-03 & 2.02 & 8.32e-03 & 2.01 & 1.12e-03 & 2.00 & 2.60e-03 & 2.00 & 3.96e-04 & 2.16 
172417 & 0.0312 & 4.83e-04 & 2.01 & 2.07e-03 & 2.01 & 2.80e-04 & 2.00 & 6.51e-04 & 2.00 & 8.99e-05 & 2.14 
688897 & 0.0156 & 1.20e-04 & 2.01 & 5.17e-04 & 2.00 & 6.99e-05 & 2.00 & 1.63e-04 & 2.00 & 2.11e-05 & 2.09 


'''
