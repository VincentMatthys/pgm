%
% x=my_chi2inv(p,nu)
%
% Finds x such that chi2cdf(x,nu)=p.  Uses chi2cdf.
%
function x=my_chi2inv(p,nu)
%
% Special cases.
%
if (p >= 1.0),
  x=+Inf;
  return;
end
if (p<=0.0),
  x=-Inf;
  return;
end
%
% Do a binary search.
%
l=0.0;
r=1.0;
while (my_chi2cdf(r,nu) < p)
  l=r;
  r=r*2;
end
%
% Now, we've got a bracket around t.
%
while (((r-l)/r) > 1.0e-5)
  m=(l+r)/2;
  if (my_chi2cdf(m,nu) > p)
    r=m;
  else
    l=m;
  end
end
x=(l+r)/2;