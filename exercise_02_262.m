clear;
load santafe;

order = 50;
% order = 50;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);


[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa','original'}, ...
   'simplex', 'crossvalidatelssvm', {10, 'mae'});

[alpha, b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'});
figure;plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'}, {alpha, b});


% Xn = Z((end-order+1):end)';
% Z(end+1) = simlssvm({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'}, {alpha, b}, Xn);

test_size = 100;
Ztrain = Z;
Zhat = predict({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'}, Ztest,length(Ztest));

figure; plot([Ztest Zhat]);

mse = sum(power((Ztest - Zhat),2)) * (1 / length(Zhat));
disp(mse);
