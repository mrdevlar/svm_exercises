% Raw Data Generation
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*X(:,1)) + 0.3.*randn(100,1);

% optFun = 'simplex';
optFun = 'gridsearch';
globalOptFun = 'csa';
% globalOptFun = 'ds';

[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel',globalOptFun}, ...
    optFun, 'crossvalidatelssvm', {10, 'mse'});

[selected, ranking] = bay_lssvmARD({X,Y,'class', gam, sig2});