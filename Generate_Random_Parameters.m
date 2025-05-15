function par=Generate_Random_Parameters(par,seed)
rng(seed);
par.S0=(par.S0_min + (par.S0_max - par.S0_min) * rand(par.D, 1))';
par.div=par.div_min + (par.div_max - par.div_min) * rand(par.D, 1);
par.K=par.K_min + (par.K_max - par.K_min) * rand(1, 1);
par.r=par.r_min + (par.r_max - par.r_min) * rand(1, 1);
par.T=par.T_min + (par.T_max - par.T_min) * rand(1, 1);
par.LB=par.L_min + (par.L_max - par.L_min) * rand(1, 1);
par.LC=par.L_min + (par.L_max - par.L_min) * rand(1, 1);
par.RB=par.R_min + (par.R_max - par.R_min) * rand(1, 1);
par.RC=par.R_min + (par.R_max - par.R_min) * rand(1, 1);
par.sF = (1 - par.RB) * par.LB;
par.CovMat = generate_covariance_matrix(par.D, par.sigma_min,par.sigma_max, par.rho_min, par.rho_max);
par.sigma=sqrt(diag(par.CovMat));
eigss=eig(par.CovMat);
while(min(eigss)<1e-6)
    par.CovMat = generate_covariance_matrix(par.D, par.sigma_min,par.sigma_max, par.rho_min, par.rho_max);
    par.sigma=sqrt(diag(par.CovMat)); 
    eigss=eig(par.CovMat);
end

try
par.CS=chol(par.CovMat,'lower');
catch
disp(eig(par.CovMat));
end

end