clear all
% load data
x = textread('EMGaussienne.data');
xtest = textread('EMGaussienne.test');
[ N d ] = size(x);
K = 4;

% random seed (make sure that everything is reproducible)
seed=1;
rand('state',seed);
randn('state',seed);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-MEANS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


max_iterations = 1000;
tolerance = 1e-10;
nrestart = 100;          % number of restarts: that many restarts are not useful in practise,
% this is simply to illustrate the variability

for irestart=1:nrestart % does several restarts

    Jold = Inf;
    rp = randperm(N);   % initialise with K random points
    mu = x(rp(1:K),:);

    for i=1:max_iterations

        % compute distances (NB: it is possible to make this faster)
        distances = zeros(N,K);
        for k=1:K
            distances(:,k) = sum( ( x - repmat(mu(k,:),N,1) ) .* ( x - repmat(mu(k,:),N,1) ) , 2 );
        end
        % E-step:
        [mindist,z]=min(distances,[],2);
        % compute distortion (dividing by N makes sure that the order of
        % magnitude remains bounded when N grows)
        J = sum( mindist ) / N;


        % fprintf('i = %d - distortion = %e\n',i,J);
        % check changes in distortion
        if J > Jold + tolerance, error('the distortion is going up!'); end % important for debugging
        if J > Jold - tolerance, break; end
        Jold = J;
        % M-step:
        for k=1:K
            mu(k,:) = mean( x(find(z==k),:) );
        end

    end
    mus{irestart} = mu;
    zs{irestart} = z;
    Js(irestart) = J;
    fprintf('irestart = %d - distortion = %e\n',irestart,J);
end

% plot histograms of all distortions
subplot(2,2,4);
hist((Js),50);
xlabel('Distortions');
title('Histogram of distortions - K-means');

% select the best distortion
[a,b] = min(Js);
mu_kmeans = mus{b};
z_kmeans = zs{b};

% plot the data
subplot(2,2,1);
colors = { 'k' 'r' 'g' 'b' };
markers = { 'o' 's' '+' 'x' };
for k=1:K
    ind = find(z_kmeans==k);
    plot( x(ind,1),x(ind,2),sprintf('%s%s',colors{k},markers{k}));
    hold on
end
hold off
title('K-means');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM - isotropic covariances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loglikold = -Inf;
mu = mu_kmeans;             % initialize with means from K-means
sigma2 =   1 * ones(K,1);     % initialize variances with large values
pis = 1/K * ones(K,1);       % initialise with uniform
max_iterations = 1000;
tolerance = 1e-10;

for i=1:max_iterations

    % compute distances ( only thing which is required for isotropic
    % covariances )
    distances = zeros(N,K);
    for k=1:K
        distances(:,k) = sum( ( x - repmat(mu(k,:),N,1) ) .* ( x - repmat(mu(k,:),N,1) ) , 2 );
    end
    % E-step: compute posterior distributions
    logtau_unnormalized = distances;
    for k=1:K
        logtau_unnormalized(:,k) = - .5 * distances(:,k) / sigma2(k) - .5 * d * log( sigma2(k) ) - .5 * d * log(2*pi) + log(pis(k));
    end
    logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
    tau = exp(logtau);

    % compute log-likelihood
    loglik = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / N;
    fprintf('i = %d - loglik = %e\n',i,loglik);
    % check changes in log likelihood
    if loglik < loglikold - tolerance, error('the distortion is going up!'); end % important for debugging

    if loglik < loglikold + tolerance, break; end
    loglikold = loglik;

    % M-step:
    for k=1:K
        mu(k,:) = sum( repmat( tau(:,k),1 ,d) .* x ) / sum( tau(:,k) );
        pis(k) = 1/N * sum(tau(:,k));
        temp = x - repmat( mu(k,:), N , 1 );
        sigma2(k) = sum(  sum( temp.^2, 2) .* tau(:,k) ) / sum( tau(:,k) ) / d;
    end
end

loglik_isotropisc = loglik  ;

% classifiy
[a,z_EM_isotropisc]=max(tau,[],2);

% plot the data
R2 = my_chi2inv(.9,2);
subplot(2,2,2);
colors = { 'k' 'r' 'g' 'b' };
markers = { 'o' 's' '+' 'x' };
for k=1:K
    ind = find(z_EM_isotropisc==k);
    plot( x(ind,1),x(ind,2),sprintf('%s%s',colors{k},markers{k}));
    hold on
    draw_ellipse_color( eye(2), - mu(k,:)', .5 * sum(mu(k,:).^2) - R2 * sigma2(k) / 2,colors{k})
end
hold off
title('EM Isotropisc');


% compute likelihood on test data
Ntest = size(xtest,1);
distances = zeros(Ntest,K);
for k=1:K
    distances(:,k) = sum( ( xtest - repmat(mu(k,:),N,1) ) .* ( xtest - repmat(mu(k,:),N,1) ) , 2 );
end
% E-step: compute posterior distributions
logtau_unnormalized = distances;
for k=1:K
    logtau_unnormalized(:,k) = - .5 * distances(:,k) / sigma2(k) - .5 * d * log( sigma2(k) )  -.5 * d * log(2*pi) + log(pis(k));
end
logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
tau = exp(logtau);

% compute log-likelihood
logliktest_isotropisc = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / Ntest ;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM - general covariances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loglikold = -Inf;
mu = mu_kmeans;                 % initialize with means from K-means
for k=1:K
    sigmas{k} =   eye(d);     % initialize digonal variances with large values
end
pis = 1/K * ones(K,1);       % initialise with uniform
max_iterations = 1000;
tolerance = 1e-10;

for i=1:max_iterations

    logtau_unnormalized = zeros(N,K);
    for k=1:K
        invSigma = inv( sigmas{k} );
        xc = ( x - repmat(mu(k,:),N,1) );
        logtau_unnormalized(:,k) = - .5 * sum( (xc * invSigma) .* xc , 2 ) - .5 * sum( log( eig( sigmas{k}) ) ) - .5 * d * log(2*pi) + log(pis(k));
    end
    logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
    tau = exp(logtau);

    % compute log-likelihood
    loglik = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / N;
    fprintf('i = %d - loglik = %e\n',i,loglik);
    % check changes in log likelihood
    if loglik < loglikold - tolerance, error('the distortion is going up!'); end % important for debugging
    if loglik < loglikold + tolerance, break; end
    loglikold = loglik;

    % M-step:
    for k=1:K
        mu(k,:) = sum( repmat( tau(:,k),1 ,d) .* x ) / sum( tau(:,k) );
        pis(k) = 1/N * sum(tau(:,k));
        temp = x - repmat( mu(k,:), N , 1 );
        sigmas{k} = 1 / sum( tau(:,k) ) * ( temp' * ( repmat( tau(:,k) ,1,d) .* temp ) );
    end
end

loglik_general = loglik ;

% classifiy
[a,z_EM_general]=max(tau,[],2);

% plot the data
R2 = my_chi2inv(.9,2);
subplot(2,2,3);
colors = { 'k' 'r' 'g' 'b' };
markers = { 'o' 's' '+' 'x' };
for k=1:K
    ind = find(z_EM_general==k);
    plot( x(ind,1),x(ind,2),sprintf('%s%s',colors{k},markers{k}));
    hold on
    draw_ellipse_color( inv(sigmas{k}) , - sigmas{k} \ ( mu(k,:)' ), .5 * mu(k,:) * ( sigmas{k} \ ( mu(k,:)' ) )  - R2/2 ,colors{k})
end
hold off
title('EM General');


% compute likelihood on test data
Ntest = size(xtest,1);
logtau_unnormalized = zeros(N,K);
for k=1:K
    invSigma = inv( sigmas{k} );
    xc = ( xtest - repmat(mu(k,:),N,1) );
    logtau_unnormalized(:,k) = - .5 * sum( (xc * invSigma) .* xc , 2 ) - .5 * sum( log( eig( sigmas{k}) ) ) - .5 * d * log(2*pi) + log(pis(k));
end
logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
tau = exp(logtau);

% compute log-likelihood
logliktest_general = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / Ntest;

% normalisee
logliktest_isotropisc
loglik_isotropisc
loglik_general
logliktest_general

% non normalisee
logliktest_isotropisc = logliktest_isotropisc*Ntest
loglik_isotropisc = loglik_isotropisc*N
loglik_general = loglik_general*N
logliktest_general= logliktest_general*Ntest


