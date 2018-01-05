function draw_ellipse_color( Q,beta,gamma,cc)
[u,e] = eig(Q);
if prod(diag(e))<=0; error('this is not an ellipse'); end
if e(1) < 0 , Q = -Q ; beta = -beta; gamma = -gamma; e = -e; end % make sure e>0
center = - Q \ beta;
c =  - beta' * center - 2 * gamma;
t = (0:100) * 2*pi/100;
x =  sqrt(c/e(1,1)) * u(1,1) * cos(t) + sqrt(c/e(2,2)) * u(1,2) * sin(t) + center(1);
y =  sqrt(c/e(1,1)) * u(2,1) * cos(t) + sqrt(c/e(2,2)) * u(2,2) * sin(t) + center(2);
plot(x,y,sprintf('%s-',cc),'linewidth',2);
