'''
Convergence test for a mixed scheme of the Navier--Stokes--Brinkman equations
The domain is (0,1)x(0,0.5)x(0,0.5)

Manufactured smooth solutions
#######################################
strong primal form: 

 eta*u - lam*div(mu*eps(u)) + grad(u)*u + grad(p) = f  in Omega
                                           div(u) = 0  in Omega 

Pure Dirichlet conditions for u 
                                                u = u_D on Gamma

Lagrange multiplier to fix the average of p
                                           int(p) = c

######################################

strong mixed form in terms of (t,sigma,u,gamma)

                 t + gamma = grad(u) 
lam*mu*t - dev(u otimes u) = dev(sigma)
        eta*u - div(sigma) = f
+ BC:
                         u = u_D on Gamma
+ trace of pressure:
 int(tr(sigma+u otimes u)) = c

FE spaces are of AFW type:



#################

NOTE: with smaller lambda, the convergence of both t and g is hindered


'''

from fenics import *
import sympy2fenics as sf
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 6

fileO = XDMFFile("outputs/out-Ex00ConvergenceIn3D-PEERS.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ****** Constant coefficients ****** #
ndim = 3
Id = Identity(ndim)
lam = Constant(0.2)

# *********** operators ****** #

epsilon = lambda vec: sym(grad(vec))
skewgr  = lambda vec: grad(vec) - epsilon(vec)
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))
curlTen = lambda ten: as_tensor([[ten[0,2].dx(1)-ten[0,1].dx(2),ten[0,0].dx(2)-ten[0,2].dx(0),ten[0,1].dx(0)-ten[0,0].dx(1)],
                                 [ten[1,2].dx(1)-ten[1,1].dx(2),ten[1,0].dx(2)-ten[1,2].dx(0),ten[1,1].dx(0)-ten[1,0].dx(1)],
                                 [ten[2,2].dx(1)-ten[2,1].dx(2),ten[2,0].dx(2)-ten[2,2].dx(0),ten[2,1].dx(0)-ten[2,0].dx(1)]])

# ******* Exact solutions for error analysis ****** #

u_str = '(sin(pi*x)*cos(pi*y)*cos(pi*z), -2*cos(pi*x)*sin(pi*y)*cos(pi*z), cos(pi*x)*cos(pi*y)*sin(pi*z))'
p_str = 'sin(x*y*z)'


# ****** Constant coefficients ****** #


nkmax = 4; l = 1

hvec = []; nvec = []; erru = []; rateu = []; errp = []; ratep = [];
errt = []; ratet = []; errsigma = []; ratesigma = []
errgamma = []; rategamma = []; 

rateu.append(0.0); ratet.append(0.0); rategamma.append(0.0);
ratep.append(0.0); ratesigma.append(0.0); 


# ***** Error analysis ***** #

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk)
    mesh = UnitCubeMesh(nps,nps,nps)
    nn   = FacetNormal(mesh)
        
    hvec.append(mesh.hmax())

    # heterogeneous viscosity 
    mu = Expression('exp(-x[0]*x[1]*x[2])', degree=3, domain = mesh)
    eta = Expression('2+sin(x[0]*x[1]*x[2])', degree=3, domain = mesh)
    
    # instantiation of exact solutions
    
    u_ex  = Expression(str2exp(u_str), degree=l+4, domain=mesh)
    p_ex  = Expression(str2exp(p_str), degree=l+4, domain=mesh)
    
    t_ex = epsilon(u_ex)
    gamma_ex = skewgr(u_ex)
    sigma_ex = lam*mu*t_ex - outer(u_ex,u_ex) - p_ex*Id
    f_ex = eta*u_ex - div(sigma_ex)
    
    # *********** Finite Element spaces ************* #
    # because of current fenics syntax, we need to define the rows
    # of sigma separately

    Hu = VectorElement('DG', mesh.ufl_cell(), l)
    Hg = VectorElement('CG', mesh.ufl_cell(), l+1, dim = 3)
    Ht = VectorElement('DG', mesh.ufl_cell(), l+ndim, dim = 8)
    Hs0 = FiniteElement('RT', mesh.ufl_cell(), l+1)
    Bub = TensorElement("B", mesh.ufl_cell(), l+ndim+1)
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)                          
    Hh = FunctionSpace(mesh, MixedElement([Ht,Hs0,Hs0,Hs0,Bub,Hu,Hg,R0]))
    
    Ph = FunctionSpace(mesh,'DG',l)
    
    print (" ****** Total DoF = ", Hh.dim())

    nvec.append(Hh.dim())
    
    # *********** Trial and test functions ********** #

    Utrial = TrialFunction(Hh)
    Usol = Function(Hh)
    t_, sig1, sig2, sig3, bsol, u, gam_, xi = split(Usol)
    s_, tau1, tau2, tau3, btest, v, delt_, ze = TestFunctions(Hh)

    t=as_tensor(((t_[0],t_[1],t_[2]),(t_[3],t_[4],t_[5]),(t_[6],t_[7],-t_[0]-t_[4])))
    s=as_tensor(((s_[0],s_[1],s_[2]),(s_[3],s_[4],s_[5]),(s_[6],s_[7],-s_[0]-s_[4])))
    
    sigma = as_tensor((sig1,sig2,sig3)) + curlTen(bsol) 
    tau   = as_tensor((tau1,tau2,tau3)) + curlTen(btest)

    gamma=as_tensor(((   0, gam_[0], gam_[1]),
                     (-gam_[0],   0, gam_[2]),
                     (-gam_[1],-gam_[2],  0)))

    delt=as_tensor(((   0, delt_[0], delt_[1]),
                     (-delt_[0],   0, delt_[2]),
                     (-delt_[1],-delt_[2],  0)))
    
    # ********** Boundary conditions ******** #

    # All Dirichlet BCs become natural in this mixed form
    
    # *************** Variational forms ***************** #

    # variational form 
    
    a   = lam*mu*inner(t,s)*dx
    b1  = - inner(sigma,s)*dx
    b   = - inner(outer(u,u),s)*dx
    b2  = inner(t,tau)*dx
    bbt = dot(u,div(tau))*dx   + inner(gamma,tau)*dx
    bb  = dot(div(sigma),v)*dx + inner(sigma,delt)*dx
    c   = eta*dot(u,v)*dx
    
    AA = a + b1 + b + b2 + bbt + bb - c 
    ZZ = tr(sigma+outer(u,u)) * ze * dx + tr(tau) * xi * dx
    FF = -dot(f_ex,v)*dx + dot(tau*nn,u_ex)*ds + tr(sigma_ex+outer(u_ex,u_ex)) * ze * dx 

    Nonl = AA + ZZ - FF
    
    Tang = derivative(Nonl, Usol, Utrial)
    problem = NonlinearVariationalProblem(Nonl, Usol, J=Tang) # In this case, no need for essential BCs
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    solver.parameters['newton_solver']['maximum_iterations'] = 25
    
    solver.solve()
    
    th_, sigh1, sigh2, sigh3, bsolh, uh, gamh_, xih = Usol.split()

    th=as_tensor(((th_[0],th_[1],th_[2]),(th_[3],th_[4],th_[5]),(th_[6],th_[7],-th_[0]-th_[4])))
    sigmah = as_tensor((sigh1,sigh2,sigh3)) + curlTen(bsolh) 
    gammah=as_tensor(((   0.0, gamh_[0], gamh_[1]),
                     (-gamh_[0],   0.0, gamh_[2]),
                     (-gamh_[1],-gamh_[2],  0.0)))
    
    # dimension-dependent
    ph = project(-1./ndim*tr(sigmah+outer(uh,uh)),Ph)
    Th = TensorFunctionSpace(mesh, 'DG', l)
    sig_v = project(sigmah, Th)
    t_v = project(th, Th)

    # saving to file

    uh.rename("u","u"); fileO.write(uh,nk*1.0)
    t_v.rename("t","t"); fileO.write(t_v,nk*1.0)
    sig_v.rename("sig","sig"); fileO.write(sig_v,nk*1.0)
    ph.rename("p","p"); fileO.write(ph,nk*1.0)

    E_t   = assemble((th-t_ex)**2*dx)
    E_sig1 = assemble((sigma_ex-sigmah)**2*dx)
    E_sig2 = assemble(dot(div(sigma_ex)-div(sigmah),div(sigma_ex)-div(sigmah))**(2./3.)*dx)# norm div,4/3
    E_u   = assemble(dot(uh-u_ex,uh-u_ex)**2*dx) # norm 0,4
    E_gam = assemble((gammah-gamma_ex)**2*dx)
    E_p   = assemble((ph-p_ex)**2*dx)
    
    errt.append(pow(E_t,0.5))
    errsigma.append(pow(E_sig1,0.5)+pow(E_sig2,0.75)) # norm div,4/3
    erru.append(pow(E_u,0.25)) # norm 0,4
    errgamma.append(pow(E_gam,0.5))
    errp.append(pow(E_p,0.5))
    
    if(nk>0):
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rategamma.append(ln(errgamma[nk]/errgamma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
        

# ********  Generating error history **** #
print('==============================================================================================================')
print('   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  ')
print('==============================================================================================================')

for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} '.format(nvec[nk], hvec[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk], erru[nk], rateu[nk], errgamma[nk], rategamma[nk], errp[nk], ratep[nk]))
print('==============================================================================================================')



'''
l = 0 

==============================================================================================================
   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  
==============================================================================================================
  1111 & 1.7321 & 3.38e+00 & 0.00 & 1.75e+01 & 0.00 & 9.86e-01 & 0.00 & 4.23e+00 & 0.00 & 1.12e+00 & 0.00 
  8698 & 0.8660 & 1.84e+00 & 0.88 & 8.32e+00 & 1.08 & 5.65e-01 & 0.80 & 1.46e+00 & 1.54 & 4.88e-01 & 1.20 
 69016 & 0.4330 & 9.77e-01 & 0.91 & 4.26e+00 & 0.97 & 3.02e-01 & 0.90 & 5.18e-01 & 1.49 & 3.52e-01 & 0.47 
550156 & 0.2165 & 4.96e-01 & 0.98 & 2.13e+00 & 1.00 & 1.55e-01 & 0.96 & 1.50e-01 & 1.79 & 1.91e-01 & 0.88 
4393876 & 0.1083 & 2.50e-01 & 0.99 & 1.07e+00 & 1.00 & 7.80e-02 & 0.99 & 5.19e-02 & 1.53 & 9.55e-02 & 1.00 

l=1

 2266 & 1.7321 & 1.72e+00 & 0.00 & 1.05e+01 & 0.00 & 5.35e-01 & 0.00 & 1.38e+00 & 0.00 & 1.00e+00 & 0.00 
 17632 & 0.8660 & 5.67e-01 & 1.60 & 2.91e+00 & 1.85 & 2.40e-01 & 1.15 & 3.93e-01 & 1.82 & 2.00e-01 & 2.32 
139372 & 0.4330 & 1.61e-01 & 1.82 & 7.57e-01 & 1.94 & 6.58e-02 & 1.87 & 1.04e-01 & 1.92 & 5.39e-02 & 1.89 
1108756 & 0.2165 & 4.43e-02 & 1.86 & 1.94e-01 & 1.97 & 1.71e-02 & 1.95 & 3.68e-02 & 1.50 & 1.41e-02 & 1.93 



'''
