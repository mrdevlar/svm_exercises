X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));

out = [15 17 19];
Y(out) = 0.7 + 0.3 * rand(size(out));
out = [41 44 46];
Y(out) = 1.5 + 0.2 * rand(size(out));

plot(X,Y);

% optFun = 'gridsearch';
% globalOptFun = 'csa';
% 
% [gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel',globalOptFun}, ...
%     optFun, 'crossvalidatelssvm', {10, 'mse'});

gam = 100;
sig2 = 0.01;

[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'});
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b});


model = initlssvm(X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess');
costFun = 'rcrossvalidatelssvm';
% wFun = 'whuber';
% wFun = 'whampel';
% wFun = 'wlogistic';
wFun = 'wmyriad';
model = tunelssvm(model, 'simplex', costFun, {10, 'mae'}, wFun);
model = robustlssvm(model);
plotlssvm(model);

