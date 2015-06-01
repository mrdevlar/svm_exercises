load iris;

X = zscore(X);
Xt= zscore(Xt);


gam   = 0.1;
sig = 2;
type  = 'c';

idx = randperm(size(X,1));

X_t = X(idx(1:80),:);
Y_t = Y(idx(1:80));
X_v = X(idx(81:100),:);
Y_v = Y(idx(81:100));


[alpha,b] = trainlssvm({X,Y,'c',gam,sig,'RBF_kernel'});
[Yt,Zt]   = simlssvm({X,Y,'c',gam,sig,'RBF_kernel'}, {alpha,b},X_v);

% Validation Data
roc(Zt,Y_v)

% Training Data Alone
roc({X,Y,'c',gam,sig,'RBF_kernel'});
