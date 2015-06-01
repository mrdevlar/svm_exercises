% Raw Data Generation
X_full = (-10:0.1:10)';
Y_full = cos(X_full) + cos(2*X_full) + 0.1 *randn(length(X_full),1);

% Training and Test Data
X = X_full(1:2:length(X_full));
Y = Y_full(1:2:length(Y_full));
Xt = X_full(2:2:length(X_full));
Yt = Y_full(2:2:length(Y_full));


% Hyperparameters
gam_list = [1, 10, 100, 1000, 10000];
sig_list = [0.00001, 0.0001, 0.001,  0.1, 1];



for gam = gam_list,
    for sig2 = sig_list,
        
        cost_crossval = crossvalidate({X,Y,'f',gam,sig2},10);
        cost_loo = leaveoneout({X,Y,'f',gam,sig2});
        
        
        output = sprintf('gamma = %f and sigma^2 = %f, Costs: crossval = %f, loo = %f', gam, sig2, cost_crossval, cost_loo);
        disp(output);
    end
end


% optFun = 'simplex';
optFun = 'gridsearch';
% globalOptFun = 'csa';
globalOptFun = 'ds';

[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel',globalOptFun}, ...
    optFun, 'crossvalidatelssvm', {10, 'mse'});
output = sprintf('gamma = %f, sigma^2 = %f, cost = %f', gam, sig2, cost);
disp(output);

[alpha, b] = trainlssvm({X,Y, 'f', gam, sig2});
plotlssvm({X,Y,'f',gam,sig2}, {alpha,b});