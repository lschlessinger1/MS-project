clear all, close all;

x = [1; 2; 3; 4; 5];                
y = [1; 2; 3; 2.1; 1.1];
xs = linspace(0, 10, 200)';

meanfunc = [];
hyp.mean = [];

covfunc = {@covPeriodic};
ell = 1.5;
p = 2.1;
sf = 2.5;
hyp.cov = log([ell; p; sf]);
K = feval(covfunc{:}, hyp.cov, x);

likfunc = @likGauss;              % Gaussian likelihood
sn = 0.1;
hyp.lik = log(sn);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, [], covfunc, likfunc, x, y);

% Negative log marginal likelihood
nlml2 = gp(hyp2, @infGaussLik, [], covfunc{:}, likfunc, x, y)

% Plot posterior
z = linspace(0, 10, 101)';
[m, s2] = gp(hyp2, @infGaussLik, [], covfunc{:}, likfunc, x, y, z);
f = [m+2*sqrt(s2); flip(m-2*sqrt(s2),1)];
fill([z; flip(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+')