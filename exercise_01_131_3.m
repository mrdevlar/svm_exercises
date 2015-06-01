load iris;

gam   = 0.1;
sig = 20;
type  = 'c';

model = {X,Y,type,[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'})


model = {X,Y,type,[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'})


model = {X,Y,type,[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'})


model = {X,Y,type,[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'})
