

%% Linear Method
load ripley;
type='c'; 
model = {Xt,Yt,type,[],[],'lin_kernel','ds'};
gam = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({Xt,Yt,type,gam,[],'lin_kernel'});
figure; plotlssvm({Xt,Yt,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xt,Yt,type,gam,[],'lin_kernel'}, {alpha,b}, X);

err = sum(Yht~=Y); 
fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

performance = crossvalidate({Xt,Yt,type,gam,[],'lin_kernel'}, 10,'misclass');
fprintf('\n On 10-Fold Crossvalidation: Error Rate = %.2f%%\n', performance*100);


roc(Zt, Y)



%% RBF Method
load ripley;
type='c'; 
model = {Xt,Yt,type,[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});


[alpha,b] = trainlssvm({Xt,Yt,type,gam,sig2,'RBF_kernel'});
figure; plotlssvm({Xt,Yt,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

[Yht, Zt2] = simlssvm({Xt,Yt,type,gam,sig2,'RBF_kernel'}, {alpha,b}, X);

err = sum(Yht~=Y); 
fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

performance = crossvalidate({Xt,Yt,type,gam,sig2,'RBF_kernel'}, 10,'misclass');
fprintf('\n On 10-Fold Crossvalidation: Error Rate = %.2f%%\n', performance*100);


roc(Zt2, Y)
