load iris;

% parameters
%gam   = 0.1;
%sig = 20;
type  = 'c';

% parameter lists
%gam_list = [0.1, 1, 10, 100];
%sig_list = [0.1, 1, 10, 20];

gam_list = [1, 10, 100];
sig_list = [0.1, 1, 10];

% Random permutation
idx = randperm(size(X,1));

% Training and Validation Sets
X_t = X(idx(1:80),:);
Y_t = Y(idx(1:80));
X_v = X(idx(81:100),:);
Y_v = Y(idx(81:100));


errlist=[];
for gam = gam_list,
    for sig = sig_list,
        
disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig)]),
% Train SVM
[alpha,b] = trainlssvm({X_t,Y_t,type,gam,sig,'RBF_kernel'});

% Predict the validation set
hat_Y_v = simlssvm({X_t,Y_t,type,gam,sig,'RBF_kernel'}, {alpha,b},X_v);

% Calculate Quantity of Errors
err = sum(hat_Y_v ~= Y_v);
errlist = [errlist, err];
% Calculate Percentage Error 
perr = err / length(Y_v) * 100;

% Fancy Print
fprintf('\n on validation: n misclassed = %d, error rate = %.2f%%\n', err, perr)
    end
end

errlist2 = vec2mat(errlist, length(gam_list));

figure;
colormap('hot');
imagesc(errlist2);
colorbar;

