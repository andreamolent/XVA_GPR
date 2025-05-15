function [polymodel,my_deg] = polyfitnH(indepvar,depvar)

%--------------------------------------------------------------------------
% "Enhancing Valuation of Variable Annuities in Lévy Models
%  with Stochastic Interest Rate"
% L. Goudenège, A. Molent, X. Wei, A. Zanette
%
% Monte Carlo pricer
% This function compute the polynomial for each sector and the optimal
% degree
% Author of the code: A. Molent (andrea.molent@uniud.it)
% Date of release: 04 April 2024
%--------------------------------------------------------------------------

%-------------------------------------------------------
% Load data parameters
%-------------------------------------------------------
M=length(depvar); % total size
M1=round(M*0.8); % trainig size


%-------------------------------------------------------
% Permute input
%------------------------------------------------------- 
data_permutation = randperm(M);
X_train=indepvar(data_permutation(1:M1),:);
Y_train=depvar(data_permutation(1:M1),:);
X_test=indepvar(data_permutation(M1+1:M),:);
Y_test=depvar(data_permutation(M1+1:M),:);
D=size(indepvar,2);

% max_deg so that no more than 1000 terms are considered
max_deg=20*(D==2)+16*(D==3)+9*(D==4)+7*(D==5)+6*(D==6)+5*(D==7);
max_deg=max_deg+4*(D==8)+4*(D==9)+3*and(D>9,D<17)+2*and(D>=18,D<50)+2*(D>=50);

%-------------------------------------------------------
% Compute optimal degree and polynomial
%-------------------------------------------------------

my_deg=0;
modelterms=Term_List(D,my_deg);
[mdl,win]=polyfitnH2(X_train,Y_train,modelterms);

if(win==0)
    fprintf("Error for deg=0. win=0");
end

Y_pred=polyvaln(mdl,X_test);
MSE_new=mean((Y_pred-Y_test).^2);

MSE_old=MSE_new+1;

while(MSE_new<MSE_old && my_deg<max_deg)
    my_deg=my_deg+1;

    if and(my_deg==2,D>50)
        modelterms=[modelterms;2*modelterms(2:end,:)];
    else

    modelterms=Term_List(D,my_deg);
    end
    MSE_old=MSE_new;

    if(2*size(modelterms,1)<M1)
        [mdl,win]=polyfitnH2(X_train,Y_train,modelterms);

        if(win==0) 
            MSE_new=MSE_new+1;
        else
        Y_pred=polyvaln(mdl,X_test);

        MSE_new=mean((Y_pred-Y_test).^2);
        end
    else
        MSE_new=MSE_new+1;
    end
end

my_deg=max(my_deg-1,0);
modelterms=Term_List(D,my_deg);
polymodel=polyfitn(indepvar,depvar,modelterms);
end