% Raw Data Generation
X_full = (-10:0.1:10)';
Y_full = cos(X_full) + cos(2*X_full) + 0.1 *randn(length(X_full),1);

% Training and Test Data
X = X_full(1:2:length(X_full));
Y = Y_full(1:2:length(Y_full));
Xt = X_full(2:2:length(X_full));
Yt = Y_full(2:2:length(Y_full));

% Hyperparameters
% gam  = 1e-6;
% sig2  = 1;

% gam_list = [1, 10, 100];
% sig_list = [0.001, 0.1, 1];


gam_list = [100, 1000, 10000];
sig_list = [0.00001, 0.0001, 0.001];

for gam = gam_list,
    for sig2 = sig_list,
        
        [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'});
        Yt_hat = simlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b}, Xt);
        %plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b});
        
        figure;
        plot(Xt,Yt,' .');
        hold on;
        plot(Xt, Yt_hat, 'r+');
        title(sprintf('LS-SVM with gamma = %.2f and sigma^2 = %.3f', gam, sig2));
        legend('Ytest', 'Yhat');
        hold off;
    end
end



