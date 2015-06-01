% Raw Data Generation
X_full = (-10:0.1:10)';
Y_full = cos(X_full) + cos(2*X_full) + 0.1 *randn(length(X_full),1);

% Training and Test Data
X = X_full(1:2:length(X_full));
Y = Y_full(1:2:length(Y_full));
Xt = X_full(2:2:length(X_full));
Yt = Y_full(2:2:length(Y_full));


gam = 100;
sig2 = 0.05;

criterion_L1 = bay_lssvm({X,Y,'f',gam,sig2},1);
criterion_L2 = bay_lssvm({X,Y,'f',gam,sig2},2);
criterion_L3 = bay_lssvm({X,Y,'f',gam,sig2},3);

sig2e = bay_errorbar({X,Y,'f',gam,sig2}, 'figure');


clear;
load iris;


% gam = 5;
% sig2=0.75;

gam_list = [0.1, 1, 5];
sig_list = [0.1, 0.4, 0.75];

for gam = gam_list,
    for sig2 = sig_list,
        bay_modoutClass({X,Y,'c', gam,sig2}, 'figure');
        
    end
end


