'''
Convergence test for a triple mixed scheme for the stationary 
Poisson-Nernst-Planck / Stokes coupled equations
Domain is (0,1)^2
Manufactured smooth solutions
Nonlinearities treated via (manual) fixed-point iterations

NOTE WE HAVE CHANGED Gk and Ck

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

fileO = XDMFFile("outputs/PNPS-Ex01Accuracy-Picard.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True


import sympy2fenics as sf
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
 
# ******* Exact solutions for error analysis ****** #
u_str   = '(cos(pi*x)*sin(pi*y),-sin(pi*x)*cos(pi*y))'
p_str   = 'x**4-y**4'
chi_str = 'sin(x)*cos(y)'
xi1_str = 'exp(-x*y)'
xi2_str = 'cos(x*y)*cos(x*y)'

# ******* Model parameters ****** #

epsilon = Constant(0.1)
kappa1 = Constant(0.25)
kappa2 = Constant(0.5)
mu     = Constant(0.01)
dim    = 2
I      = Identity(dim)

#----------------------Ranges r,s,rho,varrho  in  2D ---------------------------#
# l in [2,infty) where l and j are conjugante to each other                     #
# r   = 2*j   => s       (conjugate of r)                                       #        
# rho = 2*l   => varrho  (conjugate if rho)                                     #
# then                                                                          #
# j in (1,2] , rho in [4, infty), varrho in (1,4/3] , r in (2,4], s in [4/3,2)  #

def conjugate(t):
        tx = t/(t-1)
        return tx
    
l = Constant(2)
j = conjugate(l)
r = 2*j 
s = conjugate(r)
rho =2*l
varrho = conjugate(rho)
print("r = ",float(r),", s = ",float(s),", rho = ",float(rho),", varrho= ",float(varrho))


nkmax = 5

hh = []; dof = []; eu = []; ru = [];
esig = []; rsig = []; esig1 = []; rsig1 = [];
esig2 = []; rsig2 = []; ep = []; rp = []
echi = []; rchi = []; ephi = []; rphi = [];
exi1 = []; exi2 = []; rxi1 = []; rxi2 = [];
it =[]; etot = []; rtot = []

rtot.append(0.)
ru.append(0.0); rsig.append(0.0); rp.append(0.0)
rsig1.append(0.0); rsig2.append(0.0); rphi.append(0.0); 
rxi1.append(0.0); rxi2.append(0.0); rchi.append(0.0); 


k = 0 # polynomial degree

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)
    mesh = UnitSquareMesh(nps,nps, 'crossed')
    n = FacetNormal(mesh)
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    
    H = FiniteElement('RT', mesh.ufl_cell(), k+1) # in FEniCS RTk is understood as RT{k+1} - is a vector-valued space! 
    Q = VectorElement('DG', mesh.ufl_cell(), k)
    X = FiniteElement('RT', mesh.ufl_cell(), k+1) # also Hi's
    M = FiniteElement('DG', mesh.ufl_cell(), k) # also Qi's
    R_ = FiniteElement('R', mesh.ufl_cell(), 0) # to impose int(tr(sigma)) = constant

    Vh = FunctionSpace(mesh, MixedElement([H,H,Q,X,M,X,M,X,M,R_]))
    dof.append(Vh.dim())
    
    # ********* test and trial functions ****** #

    #H  ,H   ,Q,  X,  M,   X,   M,   X,   M,R
    taux,tauy,v,psi,lam,tau1,eta1,tau2,eta2,zeta  = TestFunctions(Vh)
    sigx,sigy,u,phi,chi,sig1,xi1,sig2,xi2  ,theta = TrialFunctions(Vh) #!

    tau = as_tensor((taux,tauy))
    sig = as_tensor((sigx,sigy))
    
    # ********* instantiation of exact solutions ****** #
    
    u_ex   = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex   = Expression(str2exp(p_str), degree=6, domain=mesh)
    chi_ex = Expression(str2exp(chi_str), degree=6, domain=mesh)
    xi1_ex = Expression(str2exp(xi1_str), degree=6, domain=mesh)
    xi2_ex = Expression(str2exp(xi2_str), degree=6, domain=mesh)

    sig_ex = mu*grad(u_ex) - p_ex*I
    phi_ex = epsilon * grad(chi_ex)
    sig1_ex = kappa1*grad(xi1_ex) + xi1_ex*kappa1/epsilon*phi_ex-xi1_ex*u_ex
    sig2_ex = kappa2*grad(xi2_ex) - xi2_ex*kappa2/epsilon*phi_ex-xi2_ex*u_ex

    # source and forcing terms
    
    ff = (xi1_ex - xi2_ex)/epsilon*phi_ex - div(sig_ex)
    gg = u_ex
    f = - div(phi_ex) - xi1_ex + xi2_ex 
    g = chi_ex
    f1 = xi1_ex - div(sig1_ex)
    f2 = xi2_ex - div(sig2_ex)
    g1 = xi1_ex; g2 = xi2_ex

    # ********* boundary conditions ******** #

    # all imposed naturally 
    
    # ********* Initial fixed-point values -- just zero ********* #

    u_k   = Function(Vh.sub(2).collapse()) #!
    phi_k = Function(Vh.sub(3).collapse()) #!
    xi1_k = Function(Vh.sub(6).collapse()) #!
    xi2_k = Function(Vh.sub(8).collapse()) #!
    
    # ********* Variational formulation for the LINEAR problem  ********* #
    
    AA  = 1./mu*inner(dev(sig),dev(tau))*dx
    BBt = dot(u,div(tau))*dx
    BB  = dot(v,div(sig))*dx
    
    FF  = dot(tau*n,gg)*ds
    GGk = (xi1_k-xi2_k)/epsilon*dot(phi_k,v)*dx - dot(ff,v)*dx

    A   = 1./epsilon*dot(phi,psi)*dx
    B1  = chi*div(psi)*dx
    B2  = lam*div(phi)*dx
    
    F   = dot(psi,n)*g*ds
    Gk  =  - f*lam*dx ######### had to make the first term bilinear

    Ai  = 1./kappa1*dot(sig1,tau1)*dx + 1./kappa2*dot(sig2,tau2)*dx
    Cit = xi1*div(tau1)*dx + xi2*div(tau2)*dx
    Ci  = eta1*div(sig1)*dx + eta2*div(sig2)*dx
    Ck  = dot(xi1_k/epsilon*phi - xi1_k/kappa1*u,tau1)*dx + dot(-xi2_k/epsilon*phi - xi2_k/kappa2*u,tau2)*dx #### had to swap the bilinearity from xi to phi/u -- is this related to the stability of Oseen linearisation?
    ## u^n \cdot \nabla u^{n+1}?

    
    Di  = xi1*eta1*dx + xi2*eta2*dx

    Fi  = dot(tau1,n)*g1*ds + dot(tau2,n)*g2*ds
    Gi  = - f1*eta1*dx - f2*eta2*dx

    lhs = AA + BBt      \
        + BB            \
        + A  + B1       \
        + B2            \
        + Ai + Cit - Ck \
        + Ci - Di       \
        + tr(sig) * zeta * dx + tr(tau) * theta * dx \
        + lam*(xi1-xi2)*dx ## moved here from Gk!!! 

    rhs = FF  \
        + GGk \
        + F   \
        + Gk  \
        + Fi  \
        + Gi  \
        + tr(sig_ex) * zeta * dx

    # ******** Picard iterations **** #
    res = 1.0; tol = 1.e-5; inc = 0; maxiter = 100

    while(res > tol and inc < maxiter):
        
        sol = Function(Vh)

        solve(lhs == rhs, sol, \
              solver_parameters = {'linear_solver':'lu'})
    
        sigx_h, sigy_h, u_h, phi_h, chi_h, sig1_h, xi1_h, sig2_h, xi2_h  ,theta_h = sol.split()

        # residual between fixed-point variables
        res = float(pow(assemble(abs(xi1_h-xi1_k)**rho*dx), 1./rho) \
                    + pow(assemble(abs(xi2_h-xi2_k)**rho*dx), 1./rho)
                    + pow(assemble(dot(phi_k-phi_h,phi_k-phi_h)**(0.5*r)*dx),1./r) \
                    + pow(assemble(abs(div(phi_k)-div(phi_h))**r*dx),1./r) \
                    + pow(assemble(((u_h-u_k)**2)**(0.5*r)*dx),1./r))

        print("iter {}, res = {}".format(inc, res))
        
        assign(u_k,u_h)
        assign(phi_k,phi_h)
        assign(xi1_k,xi1_h)
        assign(xi2_k,xi2_h)
        
        inc += 1

    it.append(inc)
    
    sig_h = as_tensor((sigx_h,sigy_h))
    Ph = FunctionSpace(mesh, 'DG', k) # trace of sigma
    p_h = project(-1./dim*tr(sig_h), Ph)
    
    u_h.rename("u","u"); fileO.write(u_h,1.*nk)
    p_h.rename("p","p"); fileO.write(p_h,1.*nk)
    chi_h.rename("chi","chi"); fileO.write(chi_h,1.*nk)
    xi1_h.rename("xi1","xi1"); fileO.write(xi1_h,1.*nk)
    xi2_h.rename("xi2","xi2"); fileO.write(xi2_h,1.*nk)
    
    # ********* Computing errors in weighted norms ****** #
    
    E_sig = pow(assemble((sig_ex-sig_h)**2*dx),0.5) \
            + pow(assemble(dot(div(sig_ex)-div(sig_h),div(sig_ex)-div(sig_h))**(0.5*s)*dx),1./s)

    E_u = pow(assemble(((u_h-u_ex)**2)**(0.5*r)*dx),1./r)

    E_p = pow(assemble((p_h-p_ex)**2*dx),0.5)

    E_phi = pow(assemble(dot(phi_ex-phi_h,phi_ex-phi_h)**(0.5*r)*dx),1./r) \
            + pow(assemble(abs(div(phi_ex)-div(phi_h))**r*dx),1./r)
    
    E_chi= pow(assemble(abs(chi_h-chi_ex)**r*dx), 1./r)
    
    E_sig1 = pow(assemble((sig1_ex-sig1_h)**2*dx),0.5) \
            + pow(assemble(abs(div(sig1_ex)-div(sig1_h))**varrho*dx),1./varrho)

    E_xi1= pow(assemble(abs(xi1_h-xi1_ex)**rho*dx), 1./rho)
    
    E_sig2 = pow(assemble((sig2_ex-sig2_h)**2*dx),0.5) \
            + pow(assemble(abs(div(sig2_ex)-div(sig2_h))**varrho*dx),1./varrho)

    E_xi2= pow(assemble(abs(xi2_h-xi2_ex)**rho*dx), 1./rho)



    esig.append(float(E_sig)); ephi.append(float(E_phi))
    esig1.append(float(E_sig1)); esig2.append(float(E_sig2))
    exi1.append(float(E_xi1)); exi2.append(float(E_xi2))
    echi.append(float(E_chi)); 
    eu.append(float(E_u)); ep.append(float(E_p))

    etot.append(float(E_sig)+float(E_u)+float(E_p)+float(E_phi)+float(E_chi)+float(E_sig1)+float(E_sig2)+float(E_xi1)+float(E_xi2))
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rtot.append(ln(etot[nk]/etot[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsig.append(ln(esig[nk]/esig[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsig1.append(ln(esig1[nk]/esig1[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsig2.append(ln(esig2[nk]/esig2[nk-1])/ln(hh[nk]/hh[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
        rphi.append(ln(ephi[nk]/ephi[nk-1])/ln(hh[nk]/hh[nk-1]))
        rchi.append(ln(echi[nk]/echi[nk-1])/ln(hh[nk]/hh[nk-1]))
        rxi1.append(ln(exi1[nk]/exi1[nk-1])/ln(hh[nk]/hh[nk-1]))
        rxi2.append(ln(exi2[nk]/exi2[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('=======================================================================')
print('  DoFs     h   e(sig) r(sig)  e(u) r(u)  e(p)  r(p)  e(phi) r(phi)  e(chi) r(chi) e(sig1) r(sig1) e(xi1) r(xi1)  e(sig2) r(sig2) e(xi2) r(xi2)  it')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d}  {:.3f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:2d}'.format(dof[nk], hh[nk], esig[nk], rsig[nk], eu[nk], ru[nk], ep[nk], rp[nk], ephi[nk], rphi[nk], echi[nk], rchi[nk], esig1[nk], rsig1[nk], exi1[nk], rxi1[nk], esig2[nk], rsig2[nk], exi2[nk], rxi2[nk], it[nk]))
print('=======================================================================')

print('=======================================================================')
print('  DoFs    h   e(tot) r(tot)  it')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d}  {:.3f} {:1.2e} {:.2f} {:2d}'.format(dof[nk], hh[nk], etot[nk], rtot[nk], it[nk]))
print('=======================================================================')



