%%% SVM Exercise 1 Script %%%

%% Two Gaussians %%

% X1 =  1 + randn(50,2);
% X2 = -1 + randn(51,2);
% 
% Y1 =  ones(50,1);
% Y2 = -ones(51,1);
% 
% X = [X1; X2];
% Y = [Y1; Y2];
% 
% figure;
% hold on;
% plot(X1(:,1), X1(:,2), 'ro');
% plot(X2(:,1), X2(:,2), 'bo');
% hold off;

%% Demos %%
democlass

help prelssvm


%% Content of Demo


%A simple example shows how to start using the toolbox for a
%classification task. We start with constructing a simple example
%dataset according to the right formatting. Data are represented 
%as matrices where each row contains one datapoint: 

X = 2.*rand(30,2)-1;
Y = sign(sin(X(:,1))+X(:,2));
X
Y


%In order to make an LS-SVM model, we need 2 extra parameters: gamma
%(gam) is the regularization parameter, determining the trade-off
%between the fitting error minimization and smoothness. In the
%common case of the RBF kernel, sigma^2 (sig2) is the bandwidth:
 
gam = 10;
sig2 = 0.2;
type = 'classification';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});


%The parameters and the variables relevant for the LS-SVM are
%passed as one cell. This cell allows for consistent default
%handling of LS-SVM parameters and syntactical grouping of related
%arguments. This definition should be used consistently throughout
%the use of that specific LS-SVM model.
%The corresponding object oriented interface
%to LS-SVMlab leads to shorter function calls (see demomodel). 
%By default, the data are preprocessed by application of the function
%prelssvm to the raw data and the function postlssvm on the
%predictions of the model. This option can explicitly be switched off in
%the call: 
 
% [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','original'});
 
%or be switched on (by default):
 
% [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'});
 
%To evaluate new points for this model, the function
%simlssvm is used:
 
Xt = 2.*rand(10,2)-1;
Ytest = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);
 
%The LS-SVM result can be displayed if the dimension of the input
%data is 2. 

plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});




%   Preprocessing of the LS-SVM
%  
%   These functions should only be called by trainlssvm or by
%   simlssvm. At first the preprocessing assigns a label to each in-
%   and output component (c for continuous, a for categorical or b
%   for binary variables). According to this label each dimension is rescaled:
%   
%       * continuous: zero mean and unit variance
%       * categorical: no preprocessing
%       * binary: labels -1 and +1
%   
%   Full syntax (only using the object oriented interface):
%   
%   >> model   = prelssvm(model)
%   >> Xp = prelssvm(model, Xt)
%   >> [empty, Yp] = prelssvm(model, [], Yt)
%   >> [Xp, Yp] = prelssvm(model, Xt, Yt)
%   
%         Outputs    
%           model : Preprocessed object oriented representation of the LS-SVM model
%           Xp    : Nt x d matrix with the preprocessed inputs of the test data
%           Yp    : Nt x d matrix with the preprocessed outputs of the test data
%         Inputs    
%           model : Object oriented representation of the LS-SVM model
%           Xt    : Nt x d matrix with the inputs of the test data to preprocess
%           Yt    : Nt x d matrix with the outputs of the test data to preprocess
%   



%% Iris Dataset %%

load iris;

% Set parameters
type='c'; 
gam = 10; 
disp('Linear kernel'),

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});

figure; plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 

fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

disp('Press any key to continue...'), pause, 



disp('Polynomial Kernel');
type='c'; 
gam = 1; 
t = 1; 
%degree = 1;

for degree=1:20,

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)
disp('Press any key to continue...'), pause,        
    
end


disp('RBF kernel and Sigma Test')
gam = 1; 
sig2list=[0.01, 0.1, 1, 5, 10, 25];

errlist=[];

for sig2=sig2list,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    figure; plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); 
    errlist = [errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    disp('Press any key to continue...'), pause,         
end

figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),





disp('RBF kernel with Regularization Test')
gamlist = [0.1, 1, 10, 10^2, 10^6]; 
sig2 = 1;

errlist=[];

for gam=gamlist,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    figure; plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); 
    errlist = [errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    disp('Press any key to continue...'), pause,         
end


figure;
plot(log(gamlist), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),








