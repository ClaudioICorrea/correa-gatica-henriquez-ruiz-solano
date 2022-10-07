'''
Convergence test for a triple mixed scheme for the stationary 
Poisson-Nernst-Planck / Stokes coupled equations
Domain is (0,1)^3
Manufactured smooth solutions
Nonlinearities treated via automatic Newton-Raphson

strong primal form: 
-----Stokes-Equations--------
-mu*laplacian(u) + grad(p) = - (xi1-xi2)*varphi/epsilon + ff in Omega 
                    div(u) = 0 in Omega
                         u = gg on Gamma
              int_Omega(p) = 0 ,
-----Poisson-Equations--------
      varphi = epsilon*grad(chi) in  Omega                       
-div(varphi) = (xi1-xi2) + f in Omega
         chi = g on Gamma ,
-----Nernst-Plack-Equations----    
for each i in {1,2}
xii - div(kappai*(grad(xii) + qi*xii*varphi/epsilon)-xii*u) = fi in  Omega
                                                         xi = gi on  Gamma
where 
qi = 1 if i = 1 and  qi = -1 if i=2

zero mean for pressure translates into a condition for the trace of sigma. Here imposed with a real Lagrange multiplier
 
'''

from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4



import sympy2fenics as sf
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
 
# ******* Exact solutions for error analysis ****** #
u_str = '((sin(pi*x))**2*sin(pi*y)*sin(2*pi*z),sin(pi*x)*(sin(pi*y))**2*sin(2*pi*z),-(sin(2*pi*x)*sin(pi*y)+sin(pi*x)*sin(2*pi*y))*(sin(pi*z))**2)'
p_str   = 'x**4-0.5*y**4-0.5*z**4'
chi_str = 'sin(x)*cos(y)*sin(z)'
xi1_str = 'exp(-x*y+z)'
xi2_str = 'cos(x*y*z)*cos(x*y*z)'

# ******* Model parameters ****** #

epsilon = Constant(0.5)
kappa1 = Constant(0.5)
kappa2 = Constant(0.25)
mu     = Constant(0.1)
dim    = 3
I = Identity(dim)


nkmax = 4

hh = []; dof = []; eu = []; ru = [];
esig = []; rsig = []; esig1 = []; rsig1 = [];
esig2 = []; rsig2 = []; ep = []; rp = []
echi = []; rchi = []; ephi = []; rphi = [];
exi1 = []; exi2 = []; rxi1 = []; rxi2 = [];

etot = []; rtot = []; mom = []; mass1 = []; energy =[]; mass2 = []

rtot.append(0.)

ru.append(0.0); rsig.append(0.0); rp.append(0.0)
rsig1.append(0.0); rsig2.append(0.0); rphi.append(0.0); 
rxi1.append(0.0); rxi2.append(0.0); rchi.append(0.0); 


for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk)
    mesh = UnitCubeMesh(nps,nps,nps)
    n = FacetNormal(mesh)
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    V = VectorElement('CG', mesh.ufl_cell(), 2)
    Q = FiniteElement('DG', mesh.ufl_cell(), 0)
    Z = FiniteElement('CG', mesh.ufl_cell(), 1) 
    R_ = FiniteElement('R', mesh.ufl_cell(), 0) # to impose p = pex 

    Vh = FunctionSpace(mesh, MixedElement([V,Q,Z,Z,Z,R_]))
    dof.append(Vh.dim())

    print("....... DOFS = ",Vh.dim())
    
    # ********* test and trial functions ****** #

    trial = TrialFunction(Vh)
    sol   = Function(Vh)

    # V  Q  Z   Z  Z   R
    v,   q, lam, eta1, eta2, zeta = TestFunctions(Vh)
    u,   p, chi, xi1,   xi2, theta = split(sol)
    
    
    # ********* instantiation of exact solutions ****** #
    
    u_ex   = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex   = Expression(str2exp(p_str), degree=6, domain=mesh)
    chi_ex = Expression(str2exp(chi_str), degree=6, domain=mesh)
    xi1_ex = Expression(str2exp(xi1_str), degree=6, domain=mesh)
    xi2_ex = Expression(str2exp(xi2_str), degree=6, domain=mesh)

    # source and forcing terms
    
    ff =  - mu*div(grad(u_ex)) + grad(p_ex) + (xi1_ex - xi2_ex)*grad(chi_ex) 
    gg = u_ex
    f = - epsilon*div(grad(chi_ex)) - (xi1_ex - xi2_ex) 
    g = chi_ex
    f1 = xi1_ex + dot(u_ex,grad(xi1_ex)) - kappa1*div(grad(xi1_ex) + xi1_ex*grad(chi_ex))
    f2 = xi2_ex + dot(u_ex,grad(xi2_ex)) - kappa2*div(grad(xi2_ex) - xi2_ex*grad(chi_ex))
    g1 = xi1_ex; g2 = xi2_ex

    # ********* boundary conditions ******** #

    bcU = DirichletBC(Vh.sub(0), gg, 'on_boundary')
    bcChi = DirichletBC(Vh.sub(2), g, 'on_boundary')
    bcXi1 = DirichletBC(Vh.sub(3), g1, 'on_boundary')
    bcXi2 = DirichletBC(Vh.sub(4), g2, 'on_boundary')
    bcs = [bcU,bcChi,bcXi1,bcXi2]
    
    
    # ********* Variational form ********* #
    
    FF =  mu * inner(grad(u),grad(v)) * dx \
        - p*div(v)*dx \
        - q*div(u)*dx \
        + (xi1-xi2)*dot(grad(chi),v) * dx \
        - dot(ff,v) * dx \
        + epsilon*dot(grad(chi),grad(lam)) * dx \
        - (xi1-xi2)*lam * dx \
        - f*lam*dx \
        + xi1*eta1*dx \
        + dot(u,grad(xi1))*eta1*dx \
        + kappa1*dot(grad(xi1) + xi1 * grad(chi), grad(eta1))*dx \
        - f1*eta1*dx\
        + xi2*eta2*dx \
        + dot(u,grad(xi2))*eta2*dx \
        + kappa2*dot(grad(xi2) - xi2 * grad(chi), grad(eta2))*dx \
        - f2*eta2*dx\
        + (p-p_ex) * zeta * dx \
        + q * theta * dx
    
    Tang = derivative(FF, sol, trial)
    problem = NonlinearVariationalProblem(FF, sol, bcs, J=Tang)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
    solver.parameters['newton_solver']['maximum_iterations'] = 25
    
    solver.solve()
    u_h, p_h, chi_h, xi1_h, xi2_h  ,theta_h = sol.split()


    Ph = FunctionSpace(mesh, 'DG', 0) # trace of sigma

    mass1_ = project(xi1_h + dot(u_h,grad(xi1_h)) - kappa1*div(grad(xi1_h) + xi1_h*grad(chi_h)) - f1,Ph)
    mass2_ = project(xi2_h + dot(u_h,grad(xi2_h)) - kappa2*div(grad(xi2_h) - xi2_h*grad(chi_h)) - f2,Ph)
    energy_ = project(epsilon*div(grad(chi_h)) + (xi1_h-xi2_h) + f,Ph)
    
    PPh = VectorFunctionSpace(mesh,'DG',0)
    mom_ = project(ff + div(mu*grad(u_ex)) - grad(p_ex) - (xi1_ex - xi2_ex)*grad(chi_ex), PPh)

    mass1.append(norm(mass1_.vector(),'linf'))
    mass2.append(norm(mass2_.vector(),'linf'))
    mom.append(norm(mom_.vector(),'linf'))
    energy.append(norm(energy_.vector(),'linf'))
    
    # ********* Computing errors in weighted norms ****** #

    E_u = errornorm(u_ex, u_h, "H1")

    E_p = pow(assemble((p_h-p_ex)**2*dx),0.5)
    
    E_chi= errornorm(chi_ex, chi_h, "H1")
    E_xi1= errornorm(xi1_ex, xi1_h, "H1")
    E_xi2= errornorm(xi2_ex, xi2_h, "H1") 
    
    etot.append(float(E_u)+float(E_p)+float(E_chi)+float(E_xi1)+float(E_xi2))
    
    if(nk>0):
        rtot.append(ln(etot[nk]/etot[nk-1])/ln(hh[nk]/hh[nk-1]))
        
        

# ********* Generating error history ****** #
print('=======================================================================')
print('  DoFs     h   e(tot) r(tot)  mom   energy  mass1  mass2  ')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d}  {:.3f} {:1.2e} {:.2f} {:1.2e} {:1.2e} {:1.2e} {:1.2e}'.format(dof[nk], hh[nk], etot[nk], rtot[nk], mom[nk], energy[nk], mass1[nk], mass2[nk]))
print('=======================================================================')


