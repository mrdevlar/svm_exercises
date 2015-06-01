A = (-10:0.05:10)';
Z = cos(A) + cos(2*A) + 0.1.*rand(size(A));


order = 5;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);

gam = 10;
sig2 = 10;
[alpha, b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'});
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b});

Xn = Z((end-order+1):end)';
Z(end+1) = simlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b}, Xn);


test_size = 100;
Ztrain = Z(1:length(Z) - test_size);
Ztest = Z(length(Z) - test_size+1:end);

horizon = length(Ztest) - order;
Zhat = predict({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, Ztest(1:order),horizon);

Ztest_w = Ztest(order+1:end);

plot([Ztest_w Zhat]);

mse = sum(power((Ztest_w - Zhat),2)) * (1 / length(Zhat));
disp(mse);
