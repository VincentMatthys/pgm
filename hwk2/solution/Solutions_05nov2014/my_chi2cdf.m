%
% p=my_chi2cdf(x,m)
%
% Computes the Chi^2 CDF, using a transformation to N(0,1) on page 333
% of Thistead, Elements of Statistical Computing.
%
% Note that x and m must be scalars.
%
function p=my_chi2cdf(x,m)
if (x == (m-1)),
  p=0.5;
else
  z=(x-m+2/3-0.08/m)*sqrt((m-1)*log((m-1)/x)+x-(m-1) )/abs(x-m+1);
  p=phi(z);
end






%
%  Calculates the normal distribution.  
%
%   z=phi(x) 
%
%  gives z=int((1/sqrt(2*pi))*exp(-t^2/2),t=-infinity..x)
%
function z=phi(x)
if (x >= 0) 
  z=.5+.5*erf(x/sqrt(2));
else
  z=1-phi(-x);
end
