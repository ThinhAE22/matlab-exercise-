function [x,y]=ellipse2D(mu,S,p)
%x and y are the coordinates of the ellipse of 2D-normal distribution 
%p=1.4/6.0/9.2 ->50/95/99% ellipse
%mu=[mux;muy] mean
%S = [sx^2,sxy;sxy,sy^2] covariance matrix 
[C,D]=eig(S); %eigenvectors and -values of S
u=C(:,1);  %first eigenvector
v=C(:,2); %second
la1=D(1,1);%first eigenvalue 
la2=D(2,2); %second
a=sqrt(p*la1); %semiaxis to u-direction 
b=sqrt(p*la2); %semiaxis to v-direction
t=0:360;
x=mu(1)+a*cosd(t)*u(1)+b*sind(t)*v(1);
y=mu(2)+a*cosd(t)*u(2)+b*sind(t)*v(2);
end