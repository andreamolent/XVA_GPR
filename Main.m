% GPR-EI and GPR-MC XVA Computation
% Implementing algorithms from "Computing XVA for American basket derivatives
% by Machine Learning techniques" (Goudenège, Molent, Zanette)

%% Clean workspace and settings
clc;
clearvars;
close all; 
warning('off', 'all');  % Suppress warnings

%% Parallel Computing Setup
n_workers = 1;                     % Number of cores for parallel computing
maxNumCompThreads(n_workers);
my_pool=Create_Pool(n_workers);

%% Simulation and Algorithm Parameters
wanna_save = false;  % Enable logging to file

%% Optional Logging
if wanna_save
    timestamp = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
    diaryFile = sprintf('log_%s.txt', timestamp);
    diary(diaryFile);
    fprintf('Logging enabled: %s\n', diaryFile);
end


%% Contract and Market Parameter Ranges
par.Type     = 'PUT_GEO';
par.D        = 2;      % Dimensionality
par.MVhat    = 0;      % 0 -> M=V       1 -> M=\hat{V}
par.M     = 1;       % Number of MC samples (MC if >10, EI otherwise) 

% GPR settings
par.P        = 125;
par.cl    = 0.01*10;    % Confidence level
par.tol   = 1e-3*10;    % Convergence tolerance



% Tree discretization parameters for computing Benchmark
par.N_CRR = 4000;    % Number of time steps for CRR benchmark
par.N     = 40;      % Number of monitoring dates


%------------------------------
% Set parameters
%------------------------------
% You can chose prefixed parameters random parameters (in ranges)

if(1) % prefixed parameters
    par.S0= 100+zeros(par.D, 1)';
    par.div=0.0+zeros(par.D, 1);
    par.K=100;
    par.r=0.03;
    par.T=1;
    par.LB=0.04;
    par.LC=0.04;
    par.RB=0.3;
    par.RC=0.3;
    par.sF = (1 - par.RB) * par.LB;
    par.sigma=0.25*ones(par.D,1);
    par.rho=0.2;
    par.CorMat = par.rho * ones(par.D)+(1-par.rho)*eye(par.D,par.D);
    par.CovMat= diag(par.sigma) * par.CorMat * diag(par.sigma);
    par.CS=chol(par.CovMat,'lower');

else % random parameters
    % Spot, Strike, Rate, Time ranges
    par.S0_min = 95;  par.S0_max = 105;
    par.K_min  = 95;  par.K_max  = 105;
    par.r_min  = 0.02; par.r_max  = 0.04;
    par.T_min  = 0.5; par.T_max  = 2;
    % Volatility and correlation
    par.sigma_min = 0.2;  par.sigma_max = 0.3;
    par.rho_min   = -0.3; par.rho_max   =  0.3;
    % Credit and liquidity spreads
    par.L_min = 0.03; par.L_max = 0.05;
    par.R_min = 0.2;  par.R_max = 0.4;
    % Dividend yield
    par.div_min = 0;    par.div_max = 0.02;
    rp= Generate_Random_Parameters(par, 1);
end

%% Benchmark Computations (CRR Tree)
if(strcmp(par.Type,'PUT_GEO'))
    fprintf('\nComputing benchmarks... \n');

    % American Method vs Benchmark Tree
    Naux=par.N;
    par.N=par.N_CRR;
    [XVA_AM, XVAh_AM, Price_rf_AM, P_AM_MeV, P_AM_MeVH] = XVA_Tree_BK(par); % American
    Price_ra_AM=par.MVhat*P_AM_MeVH+(1-par.MVhat)*P_AM_MeV;
    XVA_AM=par.MVhat*XVAh_AM+(1-par.MVhat)*XVA_AM;

    fprintf("Price_rf_AM = %.3f\t",Price_rf_AM);
    fprintf("Price_ra_AM = %.3f\t",Price_ra_AM);
    fprintf("XVA_AM = %.3f\n",par.MVhat*XVAh_AM+(1-par.MVhat)*XVA_AM);

    par.N=Naux;
    [XVA_BE, XVAh_BE, Price_rf_BE,  P_BE_MeV, P_BE_MeVH] = XVA_Tree_BK(par); % Bermudan
    Price_ra_BE=par.MVhat*P_BE_MeVH+(1-par.MVhat)*P_BE_MeV;
    XVA_BE=par.MVhat*XVAh_BE+(1-par.MVhat)*XVA_BE;

    fprintf("Price_rf_BE = %.3f\t",Price_rf_BE);
    fprintf("Price_ra_BE = %.3f\t",Price_ra_BE);
    fprintf("XVA_BE = %.3f\n\n",XVA_BE);
end

%% GPR-based XVA Computations (EI or MC)

overallStart = tic;

if par.M > 10
    fprintf('\n-- Method: GPR-MC, P = %d --\n', par.P);
else
    fprintf('\n-- Method: GPR-EI, P = %d --\n', par.P);
end


 par.MC=1e4;
    tstart = tic;

    % Choose parallel/non-parallel functions

    if par.M > 10 
        [XVA_GPR,Price_rf_GPR,Price_ra_GPR] = XVA_GPR_MC_PL( par, my_pool);  
    else 
        [XVA_GPR,Price_rf_GPR,Price_ra_GPR] = XVA_GPR_EI_PL( par, my_pool);          
    end
 
    fprintf("\nPrice_rf_GPR = %.3f\t",Price_rf_GPR);
    fprintf("Price_ra_GPR = %.3f\t",Price_ra_GPR);
    fprintf("XVA_GPR = %.3f\n",XVA_GPR);
 
    if(strcmp(par.Type,'PUT_GEO'))
    err_Price_rf_AM=100*(Price_rf_GPR-Price_rf_AM)/Price_rf_AM;
    err_Price_ra_AM=100*(Price_ra_GPR-Price_ra_AM)/Price_ra_AM;
    err_XVA_AM=100*(XVA_GPR-XVA_AM)/XVA_AM;

    err_Price_rf_BE=100*(Price_rf_GPR-Price_rf_BE)/Price_rf_BE;
    err_Price_ra_BE=100*(Price_ra_GPR-Price_ra_BE)/Price_ra_BE;
    err_XVA_BE=100*(XVA_GPR-XVA_BE)/XVA_BE;

    fprintf("\nerr_Price_rf_AM = %.2f %%\n",err_Price_rf_AM);
    fprintf("err_Price_ra_AM = %.2f %%\n",err_Price_ra_AM);
    fprintf("err_XVA_AM = %.2f %%\n",err_XVA_AM);
    fprintf("err_Price_rf_BE = %.2f %%\n",err_Price_rf_BE);
    fprintf("err_Price_ra_BE = %.2f %%\n",err_Price_ra_BE);
    fprintf("err_XVA_BE = %.2f %%\n",err_XVA_BE);
 
    end
    
totalTime = toc(overallStart);
fprintf('Total GPR computation time: %.0f sec\n', totalTime);


%% Cleanup
if wanna_save
    diary off;
end
if n_workers > 1
    delete(gcp('nocreate'));
end
fprintf('Simulation end: %s\n', datestr(now, 'dd/mm/yyyy HH:MM:SS'));
