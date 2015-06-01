clear;
load santafe;

order  = 50;
Xu     = windowize(Z,1:order+1);
Xtra   = Xu(1:end-order,1:order);
Ytra   = Xu(1:end-order,end); 
Xs     = Z(end-order+1:end,1);

[gam,sig2] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel','csa','original'},'simplex',...
'crossvalidatelssvm',{10,'mae'});

[alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel','csa','original'});

figure;plotlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel','csa','original'}, {alpha, b});

prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel','csa','original'},Xs,200);
figure;plot([prediction Ztest]);

mse = sum(power((Ztest - prediction),2)) * (1 / length(Ztest));
disp(mse);



