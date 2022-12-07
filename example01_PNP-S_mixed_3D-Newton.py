'''
Convergence test for a triple mixed scheme for the stationary 
Poisson-Nernst-Planck /  NAVIER-Stokes coupled equations
Domain is (0,1)^3
Manufactured smooth solutions
Nonlinearities treated via automatic Newton-Raphson

strong primal form: 
-----Stokes-Equations--------
lambda*u.grad(u) -mu*laplacian(u) + grad(p) = - (xi1-xi2)*varphi/epsilon + ff in Omega 
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

fileO = XDMFFile("outputs/PNPS-Ex01Accuracy-3D.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True


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

epsilon = Constant(0.1)
lmbda  = Constant(1.)
kappa1 = Constant(0.25)
kappa2 = Constant(0.5)
mu     = Constant(0.01)
dim    = 3
I = Identity(dim)

#----------------------Ranges r,s,rho,varrho  in  3D ---------------------------#
# l = 3  where l and j are conjugante to each other                             #
# r   = 2*j   => s       (conjugate of r)                                       #        
# rho = 2*l   => varrho  (conjugate if rho)                                     #
# then                                                                          #
# j = 3/2 , rho = 6, varrho =6/5, r= 3 and s = 3/2                              #

def conjugate(t):
        tx = t/(t-1)
        return tx

l = Constant(3)
j = conjugate(l)
r = 2*j 
s = conjugate(r)
rho =2*l
varrho = conjugate(rho)
print("r = ",float(r),", s = ",float(s),", rho = ",float(rho),", varrho= ",float(varrho))

nkmax = 5

hh = []; dof = []; niters = []; eu = []; ru = []; et = []; rt = []; 
esig = []; rsig = []; esig1 = []; rsig1 = [];
esig2 = []; rsig2 = []; ep = []; rp = []
echi = []; rchi = []; ephi = []; rphi = [];
exi1 = []; exi2 = []; rxi1 = []; rxi2 = [];


etot = []; rtot = []; mom = []; ene = []; mass1 =[]; mass2 =[] 

rtot.append(0.); rt.append(0)

ru.append(0.0); rsig.append(0.0); rp.append(0.0)
rsig1.append(0.0); rsig2.append(0.0); rphi.append(0.0); 
rxi1.append(0.0); rxi2.append(0.0); rchi.append(0.0); 
# polynomial degree

k = 0

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk)
    mesh = UnitCubeMesh(nps,nps,nps)
    n = FacetNormal(mesh)
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    # Navier-Stokes-problem 
    Hu = VectorElement('DG', mesh.ufl_cell(), k)         # H1 -> (u,v)
    Htt= VectorElement('DG', mesh.ufl_cell(), k, dim=8)  # H2 -> (t,s)
    Q = FiniteElement('RT', mesh.ufl_cell(), k+1)        # Q -> (sigma,tau)
    # PNP problem
    X = FiniteElement('RT', mesh.ufl_cell(), k+1) # also Hi's 
    M = FiniteElement('DG', mesh.ufl_cell(), k) # also Qi's
    # to impose int(tr(sigma)) = constant
    R_ = FiniteElement('R', mesh.ufl_cell(), 0) 

    Vh = FunctionSpace(mesh, MixedElement([Hu,Htt,Q,Q,Q,X,M,X,M,X,M,R_]))
    dof.append(Vh.dim())

    print("....... DOFS = ",Vh.dim())
    
    # ********* test and trial functions ****** #

    trial = TrialFunction(Vh)
    sol   = Function(Vh)

  #Hu, Htt,    Q,    Q,    Q,   X,   M,    X,    M,    X,   M,   R
    v,  s_, taux, tauy, tauz, psi, lam, tau1, eta1, tau2, eta2,  zeta  = TestFunctions(Vh)
    u,  t_, sigx, sigy, sigz, phi, chi, sig1,  xi1, sig2,  xi2,  theta = split(sol)


    tt = as_tensor(((t_[0],t_[1],t_[2]),(t_[3],t_[4],t_[5]),(t_[6],t_[7],-t_[0]-t_[4])))
    ss = as_tensor(((s_[0],s_[1],s_[2]),(s_[3],s_[4],s_[5]),(s_[6],s_[7],-s_[0]-s_[4])))
    tau = as_tensor((taux,tauy,tauz))
    sig = as_tensor((sigx,sigy,sigz))
    
    # ********* instantiation of exact solutions ****** #
    
    u_ex   = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex   = Expression(str2exp(p_str), degree=6, domain=mesh)
    chi_ex = Expression(str2exp(chi_str), degree=6, domain=mesh)
    xi1_ex = Expression(str2exp(xi1_str), degree=6, domain=mesh)
    xi2_ex = Expression(str2exp(xi2_str), degree=6, domain=mesh)

    tt_ex  = grad(u_ex)
    sig_ex = mu*tt_ex - lmbda*0.5*outer(u_ex,u_ex) - p_ex*I
    phi_ex = epsilon * grad(chi_ex)
    sig1_ex = kappa1*grad(xi1_ex) + xi1_ex*kappa1/epsilon*phi_ex-xi1_ex*u_ex
    sig2_ex = kappa2*grad(xi2_ex) - xi2_ex*kappa2/epsilon*phi_ex-xi2_ex*u_ex

    # source and forcing terms
    
    ff = (xi1_ex - xi2_ex)/epsilon*phi_ex - div(sig_ex) + 0.5*lmbda*tt_ex*u_ex 
    gg = u_ex
    f = - div(phi_ex) - xi1_ex + xi2_ex 
    g = chi_ex
    f1 = xi1_ex - div(sig1_ex)
    f2 = xi2_ex - div(sig2_ex)
    g1 = xi1_ex; g2 = xi2_ex

    # ********* boundary conditions ******** #

    # all imposed naturally 
    
    # ********* Variational form ********* #
    
    # ---- NS ----- # 
    aa_uv  = mu*inner(tt,ss)*dx
    cc_uuv = 0.5*lmbda*(dot(tt*u,v) - inner(outer(u,u),ss))*dx
    bb_vsi = - inner(sig,ss)*dx - dot(v,div(sig))*dx
    bb_uta = - inner(tt,tau)*dx - dot(u,div(tau))*dx

    FF_v   = (xi2-xi1)/epsilon*dot(phi,v)*dx + dot(ff,v)*dx 
    GG_tau = -dot(tau*n,gg)*ds

    # ---- Poisson ----- # 
    a_phipsi  = 1./epsilon*dot(phi,psi)*dx
    b1_psichi = chi*div(psi)*dx
    b2_philam = lam*div(phi)*dx
    F_psi     = dot(psi,n)*g*ds
    G_lam     = - lam*(xi1-xi2)*dx - f*lam*dx

    # ---- Nernst-Planck ----- #
    ai_st  = 1./kappa1*dot(sig1,tau1)*dx + 1./kappa2*dot(sig2,tau2)*dx
    ci_tx  = xi1*div(tau1)*dx + xi2*div(tau2)*dx
    ci_se  = eta1*div(sig1)*dx + eta2*div(sig2)*dx
    cpu_tx = dot(xi1/epsilon*phi - xi1/kappa1*u,tau1)*dx + dot(-xi2/epsilon*phi - xi2/kappa2*u,tau2)*dx
    di_xe  = xi1*eta1*dx + xi2*eta2*dx
    Fi_t   = dot(tau1,n)*g1*ds + dot(tau2,n)*g2*ds 
    Gi_e   = - f1*eta1*dx - f2*eta2*dx

    # ---- Lagrange multiplier to impose trace of sigma --- # 
    ZZ = (tr(sig + lmbda*0.5*outer(u,u))- tr(sig_ex + lmbda*0.5*outer(u_ex,u_ex))) * zeta * dx \
        + tr(tau) * theta * dx
    
    # global nonlinear variational form 
    Nonl = aa_uv + cc_uuv + bb_vsi  - FF_v   \
         + bb_uta                   - GG_tau \
         + a_phipsi + b1_psichi     - F_psi  \
         + b2_philam                - G_lam  \
         + ai_st + ci_tx - cpu_tx   - Fi_t   \
         + ci_se - di_xe            - Gi_e   \
         + ZZ 
         
    Tang = derivative(Nonl, sol, trial)
    #Tang = derivative(FF, sol, trial)
    problem = NonlinearVariationalProblem(Nonl, sol, J=Tang)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'umfpack'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
    solver.parameters['newton_solver']['maximum_iterations'] = 25
    
    solver.solve()
    #niters.append(niters_[0])
    
    u_h, th_, sigx_h, sigy_h, sigz_h, phi_h, chi_h, sig1_h, xi1_h, sig2_h, xi2_h  ,theta_h = sol.split()

    tt_h = as_tensor(((th_[0],th_[2],th_[3]),(th_[4],th_[1],th_[5]),(th_[6],th_[7],-th_[0]-th_[1])))

    sig_h = as_tensor((sigx_h,sigy_h,sigz_h))

    Ph = FunctionSpace(mesh, 'DG', k) # trace of sigma

    p_h = project(-1./dim*tr(sig_h + 0.5*lmbda*outer(u_h,u_h)), Ph)
    
    u_h.rename("u","u"); fileO.write(u_h,1.*nk)
    p_h.rename("p","p"); fileO.write(p_h,1.*nk)
    chi_h.rename("chi","chi"); fileO.write(chi_h,1.*nk)
    xi1_h.rename("xi1","xi1"); fileO.write(xi1_h,1.*nk)
    xi2_h.rename("xi2","xi2"); fileO.write(xi2_h,1.*nk)

    Th = TensorFunctionSpace(mesh, 'DG', k)
    tt_post = project(tt_h,Th)
    sig_post = project(sig_h,Th)

    tt_post.rename("tt","tt"); fileO.write(tt_post,1.*nk)
    sig_post.rename("sig","sig"); fileO.write(sig_post,1.*nk)

    phi_h.rename("phi","phi"); fileO.write(phi_h,1.*nk)
    sig1_h.rename("sig1","sig1"); fileO.write(sig1_h,1.*nk)
    sig2_h.rename("sig2","sig2"); fileO.write(sig2_h,1.*nk)
    
    # ********* Computing errors in weighted norms ****** #
    
    E_sig = pow(assemble((sig_ex-sig_h)**2*dx),0.5) \
            + pow(assemble(dot(div(sig_ex)-div(sig_h),div(sig_ex)-div(sig_h))**(0.5*s)*dx),1./s)

    E_t = pow(assemble((tt_h-tt_ex)**2*dx),0.5)
    
    ######
    E_u = pow(assemble(((u_h-u_ex)**2)**(0.5*4)*dx),1./4)

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
 



    esig.append(float(E_sig)); et.append(float(E_t));
    ephi.append(float(E_phi))
    esig1.append(float(E_sig1)); esig2.append(float(E_sig2))
    exi1.append(float(E_xi1)); exi2.append(float(E_xi2))
    echi.append(float(E_chi)); 
    eu.append(float(E_u)); ep.append(float(E_p))
    etot.append(float(E_t) + float(E_sig)+float(E_u)+float(E_p)+float(E_phi)+float(E_chi)+float(E_sig1)+float(E_sig2)+float(E_xi1)+float(E_xi2))
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rt.append(ln(et[nk]/et[nk-1])/ln(hh[nk]/hh[nk-1]))
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
print('  DoFs     h    e(u) r(u)   e(t) r(t)  e(sig) r(sig)   e(p)  r(p)  e(phi) r(phi)  e(chi) r(chi) e(sig1) r(sig1) e(xi1) r(xi1)  e(sig2) r(sig2) e(xi2) r(xi2)  ')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d}  {:.3f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f}'.format(dof[nk], hh[nk], eu[nk], ru[nk], et[nk], rt[nk], esig[nk], rsig[nk],  ep[nk], rp[nk], ephi[nk], rphi[nk], echi[nk], rchi[nk], esig1[nk], rsig1[nk], exi1[nk], rxi1[nk], esig2[nk], rsig2[nk], exi2[nk], rxi2[nk]))
print('=======================================================================')


'''
print('=======================================================================')
print('  DoFs    h   e(tot) r(tot)  mom_h it')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d}  {:.3f} {:1.2e} {:.2f} {:1.2e} {:1.2e} {:1.2e} {:1.2e} {:2d}'.format(dof[nk], hh[nk], etot[nk], rtot[nk], mom[nk], mass1[nk], mass2[nk], ene[nk], niters[nk]))
print('=======================================================================')
'''

'''


    esig.append(float(E_sig)); ephi.append(float(E_phi))
    esig1.append(float(E_sig1)); esig2.append(float(E_sig2))
    exi1.append(float(E_xi1)); exi2.append(float(E_xi2))
    echi.append(float(E_chi)); 
    eu.append(float(E_u)); ep.append(float(E_p))
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
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
print('  DoFs     h   e(sig) r(sig)  e(u) r(u)  e(p)  r(p)  e(phi) r(phi)  e(chi) r(chi) e(sig1) r(sig1) e(xi1) r(xi1)  e(sig2) r(sig2) e(xi2) r(xi2)  ')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d}  {:.3f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f} {:1.2e} {:.2f}'.format(dof[nk], hh[nk], esig[nk], rsig[nk], eu[nk], ru[nk], ep[nk], rp[nk], ephi[nk], rphi[nk], echi[nk], rchi[nk], esig1[nk], rsig1[nk], exi1[nk], rxi1[nk], esig2[nk], rsig2[nk], exi2[nk], rxi2[nk]))
print('=======================================================================')

'''

