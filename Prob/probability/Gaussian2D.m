%% 2D-normal distribution 
clear
close all
%mean
mux=1
muy=1
mu=[mux;muy]
%standard deviations
sx=1.5
sy=1.5
%correlation coefficient
rho=-0.3
%covariance
sxy=rho*sx*sy
%covariance matix
S=[sx^2,sxy
   sxy,sy^2]
%50/95/99% ellipses


%function-file ellipse2D.m in the current folder 
p1=1.4
[x1,y1]=ellipse2D(mu,S,p1); % 50%
p2=6.0
[x2,y2]=ellipse2D(mu,S,p2); % 95%
p3=9.2
[x3,y3]=ellipse2D(mu,S,p3); % 99%



figure(1)
plot(x1,y1,'r','linewidth',1.5)
hold
plot(x2,y2,'g','linewidth',1.5)
plot(x3,y3,'b','linewidth',1.5)
plot(mu(1),mu(2),'k.','markersize',25)
hold off
axis equal
grid
xlabel('x')
ylabel('y','rotation',0)
title(['\sigma_x = ',num2str(sx),', \sigma_y = ',num2str(sy),...
        ', \sigma_{xy} = ',num2str(sxy),', \rho = ',num2str(rho)])
legend('50 %','95 %','99 %')
%% surface z=f(X)
%xy-pairs
a=4*max([S(1,1),S(2,2)]);
xx=(mux-a):a/50:(mux+a);
yy=(muy-a):a/50:(muy+a);
[xp,yp]=meshgrid(xx,yy);



g=1/(2*pi*sqrt(det(S)));
f= @(X) g*exp(-1/2*(X-mu)'/S*(X-mu));%A/B=A*B^-1

n=length(xx);
ff=zeros(n,n);
for r=1:n
    for s=1:n
       Xrs=[xp(r,s);yp(r,s)]; 
       ff(r,s)=f(Xrs);
    end
end

N=length(x1);
%z=f(X) above the ellipses
f50=g*exp(-1/2*p1)*ones(1,N);
f95=g*exp(-1/2*p2)*ones(1,N);
f99=g*exp(-1/2*p3)*ones(1,N);


figure(2)
surf(xp,yp,ff)
hold
p50=plot3(x1,y1,f50,'r','linewidth',2)
p95=plot3(x2,y2,f95,'g','linewidth',2)
p99=plot3(x3,y3,f99,'b','linewidth',2)
hold off
grid on

xlabel('x')
ylabel('y')
zlabel('z = f(X)')
title(['\sigma_x = ',num2str(sx),', \sigma_y = ',num2str(sy),...
        ', \sigma_{xy} = ',num2str(sxy),', \rho = ',num2str(rho)])
legend([p50,p95,p99],{'50 %','95 %','99 %'})
