%% Linear Case

load diabetes;

X  = trainset;
Y  = labels_train;
Xt = testset;
Yt = labels_test;

type='c'; 
model = {X,Y,type,[],[],'lin_kernel','ds'};
gam = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
%figure; plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel', 'preprocess'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

performance = crossvalidate({X,Y,type,gam,[],'lin_kernel'}, 10,'misclass');
fprintf('\n On 10-Fold Crossvalidation: Error Rate = %.2f%%\n', performance*100);


roc(Zt, Yt)

%% RBF Case

load diabetes;

X  = trainset;
Y  = labels_train;
Xt = testset;
Yt = labels_test;

type='c'; 
model = {X,Y,type,[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});


[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
%figure; plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

[Yht, Zt2] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel', 'preprocess'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

performance = crossvalidate({X,Y,type,gam,sig2,'RBF_kernel'}, 10,'misclass');
fprintf('\n On 10-Fold Crossvalidation: Error Rate = %.2f%%\n', performance*100);


roc(Zt2, Yt)