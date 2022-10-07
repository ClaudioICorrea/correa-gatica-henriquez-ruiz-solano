

clear all; close all;

M0 = [
  145  1.732 1.60e+00 0.00 2.79e+00 0.00 2.34e-01 0.00 1.96e-01 0.00 1.37e-01 0.00 4.99e+00 0.00 4.94e-01 0.00 3.34e+00 0.00 1.88e-01 0.00
  1009  0.866 9.21e-01 0.80 1.76e+00 0.66 1.56e-01 0.59 1.25e-01 0.42 7.23e-02 0.92 2.44e+00 1.03 3.24e-01 0.61 1.51e+00 1.15 1.34e-01 0.49
  7489  0.433 5.11e-01 0.85 7.27e-01 1.28 8.78e-02 0.83 6.22e-02 1.00 3.67e-02 0.98 1.11e+00 1.14 1.65e-01 0.98 6.54e-01 1.20 7.49e-02 0.84
 57601  0.217 2.61e-01 0.97 2.49e-01 1.54 4.46e-02 0.98 2.26e-02 1.46 1.80e-02 1.03 4.46e-01 1.31 7.11e-02 1.21 2.51e-01 1.38 3.67e-02 1.03
451585  0.108 1.30e-01 1.00 9.26e-02 1.43 2.18e-02 1.03 7.29e-03 1.63 8.90e-03 1.02 1.88e-01 1.25 3.13e-02 1.18 1.02e-01 1.30 1.83e-02 1.00
3575809 0.0505 6.5e-2   1    4.5e-2   1.3  1.09e-2  1    2.5e-3  1.5  4.45e-3  1    9.1e-2  1.2   1.55e-2  1    5.0e-2   1    9.1e-3  1];

n  = M0(:,1);  
h  = M0(:,2);  
es = M0(:,3); 
eu = M0(:,5);
ep = M0(:,7);
ephi = M0(:,9);
echi = M0(:,11);
es1 = M0(:,13);
exi1 = M0(:,15);
es2 = M0(:,17);
exi2 = M0(:,19);

HH=figure('Position',[429 254 550 550]);
hh1=axes('Position',[0.11 0.1 0.86 0.87]);

G1=loglog(n,h,'ko-',n,eu,'^--',n,ep,'*--',n,echi,'d--',n,exi1,'v-.',n,exi2,'+-.');
xlabel('DoF','Fontsize',20,'Interpreter','LaTex');
leg1=legend({'$O(h)$','$\|\textbf{u}-\textbf{u}_h\|_{\mathbf{Q}}$','$\|p-p_h\|_{0,\Omega}$','$\|\chi-\chi_h\|_{M_1}$','$\|\xi_{1}-\xi_{1,h}\|_{\mathrm{Q}_1}$', '$\|\xi_{2}-\xi_{2,h}\|_{\mathrm{Q}_2}$'},'Location','SouthWest','Interpreter','LaTex','Fontsize',20);
axis tight; 
set(gca,'Linewidth',3);
set(gca,'GridAlpha',0.1);
set(G1,'Markersize',15);
set(gca,'Fontsize',20);
set(gca,'TickLabelInterpreter','LaTex')
set(G1,'Linewidth',2.5);



%	     '$\texttt{e}(\mbox{\boldmath$\varphi$})$','$\texttt{e}(\chi)$','$\texttt{e}(\mbox{\boldmath$\sigma$}_1)$','$\texttt{e}(\xi_1)$','$\texttt{e}(\mbox{\boldmath$\sigma$}_2)$','$\texttt{e}(\xi_2)$'},'Location','SouthWest','Interpreter','LaTex','Fontsize',20);


HH=figure('Position',[429 600 550 550]);
hh2=axes('Position',[0.11 0.1 0.86 0.87]);
G2=loglog(n,h,'ko-',n,es,'^--',n,ephi,'*--',n,es1,'v-.',n,es2,'+-.');
xlabel('DoF','Fontsize',20,'Interpreter','LaTex');
leg1=legend({'$O(h)$','$\|\mbox{\boldmath$\sigma$}-\mbox{\boldmath$\sigma$}_h\|_{\mathbf{H}}$','$\|\mbox{\boldmath$\varphi$}-\mbox{\boldmath$\varphi$}_h\|_{X_2}$','$\|\mbox{\boldmath$\sigma$}_1-\mbox{\boldmath$\sigma$}_{1,h}\|_{\mathrm{H}_1}$','$\|\mbox{\boldmath$\sigma$}_2-\mbox{\boldmath$\sigma$}_{2,h}\|_{\mathrm{H}_2}$'},'Location','SouthWest','Interpreter','LaTex','Fontsize',20);
axis tight;


%G=loglog(n,h,'ko-',n,es,'v--',n,eu,'^--',n,ep,'*--',n,ephi,'+--',n,echi,'d--',n,es1,'v-.',n,exi1,'^-.',n,es2,'*-.',n,exi2,'+-.');
%xlh = xlabel('DoF','Fontsize',20,'Interpreter','LaTex');
%,'Position',[Xlb Ylb]);
%set(xlh,'position',[184.8736 0.0003 -1.0000])

%leg=legend({'$O(h)$','$\texttt{e}(\mbox{\boldmath$\sigma$})$','$\texttt{e}(\textbf{u})$','$\texttt{e}(p)$','$\texttt{e}(\mbox{\boldmath$\varphi$})$','$\texttt{e}(\chi)$','$\texttt{e}(\mbox{\boldmath$\sigma$}_1)$','$\texttt{e}(\xi_1)$','$\texttt{e}(\mbox{\boldmath$\sigma$}_2)$','$\texttt{e}(\xi_2)$'},'Location','SouthWest','Interpreter','LaTex','Fontsize',20);
%set(leg.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.0;.0;.0;.0]));
%grid on;

set(gca,'Linewidth',3);
set(gca,'GridAlpha',0.1);
set(G2,'Markersize',15);
set(gca,'Fontsize',20);
set(gca,'TickLabelInterpreter','LaTex')
set(G2,'Linewidth',2.5);
