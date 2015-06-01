load iris;

gam   = 0.1;
sig = 20;
type  = 'c';

% parameter lists
gam_list = [0.1, 1, 10, 100];
sig_list = [0.1, 1, 10, 20];



performance = crossvalidate({X,Y,type,gam,sig,'RBF_kernel'}, 10,'misclass');

performance2 = leaveoneout({X,Y,type,gam,sig,'RBF_kernel'});


