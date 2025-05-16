function [XVA,Price_rf,Price_ra]=XVA_GPR_EI_PL(par,my_pool)
% XVA pricer with GPR‑EI approach
% Code author: Molent Andrea
% Creation: 19 May 2022
% Last update: 14 May 2025
%-------------------------------------------------------------------------%
%  This function prices the total valuation adjustment (XVA) for a variety
%  of exotic pay‑offs using a Gaussian‑Process‑Regression / Exact‑
%  Integration (GPR‑EI) algorithm, as described in Molent (2022).
%
%  ----------------------------------------------------------------------
%  INPUT ARGUMENTS
%  ----------------------------------------------------------------------
%  par      : structure with all model, market and contract parameters
%  my_pool  : handle to an existing MATLAB parallel pool (parpool object)
%
%  ----------------------------------------------------------------------
%  OUTPUT ARGUMENTS
%  ----------------------------------------------------------------------
%  XVA      : Total valuation adjustment  = Price_rf  −  Price_ra
%  Price_rf : Option price under the American exercise feature (risk‑free)
%  Price_ra  : Option price under the hedging measure (collateralised)
%
%  ----------------------------------------------------------------------
%  INTERNAL WORK‑FLOW
%  ----------------------------------------------------------------------
%   1. Unpack parameters and map the option type to a numeric code.
%   2. Pre‑compute constants that will be reused many times.
%   3. Build a low‑discrepancy Halton grid for the regression design‑set.
%   4. Simulate the state variables on that grid and compute intrinsic
%      values for each time step.
%   5. Estimate the European component with a control‑variates technique.
%   6. Perform a backward dynamic‑programming recursion, where the
%      conditional expectations are computed via Gaussian Process
%      Regression (fitrgp) – one model for the risk‑free measure and one
%      for the risky measure.
%   7. Finally, collect all results and return XVA, Price_rf and Price_ra.
%-------------------------------------------------------------------------%

rng(1); % Set random seed for reproducibility


%% ---------------------------------------------------------------------
%% 1. UNPACK INPUTS
%% ---------------------------------------------------------------------
S0      = par.S0;      % Spot vector (1 × D)
K       = par.K;       % Strike (scalar)
r       = par.r;       % Risk‑free rate (continuously compounded)
div     = par.div;     % Dividend / funding spread vector (1 × D)
CovMat  = par.CovMat;   % Variance covariance matrix  (D × D)
sigma   = par.sigma;   % Vol‑vector (1 × D)
CS      = par.CS;      % Cholesky factor of Brownian covariance (D × D)
T       = par.T;       % Maturity (years)
LB      = par.LB;      % Lender's borrowing spread
LC      = par.LC;      % Lender's collateral spread
RB      = par.RB;      % Receiver's borrowing spread
RC      = par.RC;      % Receiver's collateral spread
sF      = par.sF;      % Funding spread
Type    = par.Type;    % Pay‑off identifier (string)
MVhat   = par.MVhat;   % Flag: 1 ⇒ M = \hat V,   0 ⇒ M = V
P       = par.P;       % Number of regression points (design set)
N       = par.N;       % Number of time steps in the tree / MC grid
D       = par.D;       % Number of underlying assets / dimensions
cl      = par.cl;      % Confidence level for MC error control
tol     = par.tol;     % Absolute tolerance for MC integration

% NOTE: my_pool is used only inside parallel loops (parfor).

%% ---------------------------------------------------------------------
%% 2. MAP OPTION TYPE TO PAY‑OFF FUNCTION
%% ---------------------------------------------------------------------
% A numeric code speeds‑up later branching in parfor‑loops.  In addition
% a function handle (payoff_fun) is created so that the intrinsic value
% can be evaluated on matrices of simulated prices with a single call.

if(strcmp(Type,'PUT_GEO'))
    num_type = 1;
    payoff_fun = @(X_) max(K - geomean(X_,2), 0);
elseif(strcmp(Type,'PUT_ARI'))
    num_type = 2;
    payoff_fun = @(X_) max(K - mean(X_,2), 0);
elseif(strcmp(Type,'CALL_MAX'))
    num_type = 3;
    payoff_fun = @(X_) max(max(X_,[],2) - K, 0);
elseif(strcmp(Type,'PTF_SWAP'))
    % Portfolio swap: long average of first D/2 assets, short average of the
    % remaining ones, with strike K.
    num_type = 4;
    D1 = D/2; D2 = D1 + 1;
    payoff_fun = @(X_) max(mean(X_(:,1:D1),2) - mean(X_(:,D2:end),2), K);
else
    error('XVA_GPR_EI_PL:UnknownType', ...
          'Unsupported option type "%s".', Type);
end

%% ---------------------------------------------------------------------
%% 3. PRE‑COMPUTE CONSTANTS (used many times inside loops)
%% ---------------------------------------------------------------------
% Collateral adjusted discount factors and branching coefficients.

r0  = r + LB + LC;           % Funding‑adjusted rate

dt  = T / N;                 % Time step
dtm = 0.5 * dt;              % Half time step (used for theta‑schemes)

% Discount factor under risk‑free and risky measures
% (risk‑free: df, risky: df0)

df  = exp(-r  * dt);
cf  = 1/df;                  % Convenience factor (pre‑computed reciprocal)
df0 = exp(-r0 * dt);

% Collateral coefficients for positive (cp) and negative (cm) exposures
cp = LB + LC * RC - sF;
cm = LC + LB * RB;

% Denominators used when M = \hat V (positive / negative branches)
dp = 1 / (1 - dtm * cp);
dm = 1 / (1 - dtm * cm);

% Monte‑Carlo settings: initial and maximum #paths (adapted to dimension D)
% The adaptive MC routine starts from nMC0 paths and increases up to
% nMC_max when the CI width is above the prescribed tolerance.

nMC0    = 1e3;                           % initial number of paths
nMC_max = max(1e5, 4e7 / D);            % dimension‑adjusted upper bound
if(num_type == 4)                       % swap pay‑off is slower ⇒ fewer paths
    nMC_max = nMC_max / 4;
end

%% ---------------------------------------------------------------------
%% 4. LOW‑DISCREPANCY SEQUENCE (Halton) FOR THE DESIGN SET
%% ---------------------------------------------------------------------
% Halton points are used to build a quasi‑random grid (size P × D) where
% GPR models are trained.  Skip / leap parameters ensure better uniformity.

prime_num = primes(1000);
if(D < 60)
    Halton = haltonset(D,'Skip',1e4,'Leap',prime_num(D+1)-1);
else
    Halton = haltonset(D,'Skip',1e4);
end

Halton_points = net(Halton, P-1);

% Convert Halton uniform samples into Brownian increments via inverse CDF
pd = makedist('Normal');
G0 = (CS * [zeros(1,D); icdf(pd, Halton_points)]')';  % (P × D)

%% ---------------------------------------------------------------------
%% 5. ALLOCATE MAIN CONTAINERS
%% ---------------------------------------------------------------------
G  = zeros(P, D, N);     % Brownian increments for each step
hS = zeros(P, N);        % Intrinsic values (pay‑off) per path / step
XX = zeros(P, D, N);     % Simulated price paths

%% ---------------------------------------------------------------------
%% 6. FORWARD SIMULATION OF PATHS AND INTRINSIC VALUES
%% ---------------------------------------------------------------------
for n = 1:N
    tn = n * dt;                     % current time

    % Brownian part (scaled by sqrt(t) or sqrt(T) depending on D)
    if(D < 10)
        Gn = sqrt(T) * G0;           % time‑independent trick for low dims
    else
        Gn = sqrt(tn) * G0;          % standard Brownian scaling
    end

    % Underlying price dynamics: log‑normal with drift (r‑div-½σ²)
    Xn = S0 .* exp(tn*((r - div - 0.5 * sigma.^2))' + Gn);

    % Store state, Brownian increment and intrinsic value
    XX(:,:,n) = Xn;
    G(:,:,n)  = Gn;
    hS(:,n)   = payoff_fun(Xn);
end

% Intrinsic value at time 0 (needed for the last backward step)
EV0 = payoff_fun(S0);

%% ---------------------------------------------------------------------
%% 7. EUROPEAN COMPONENT (GPR‑controlled MC)
%% ---------------------------------------------------------------------
fprintf(" EU step ");
eu_tic = tic();

ci_fac = norminv(1 - cl/2);       % Z‑score for the CI at confidence cl
CSEU   = CS * randn(D, nMC_max);   % Standard normals for antithetic MC

% -------- Terminal payoff (t = T) -------------
XnF_A  = S0 .* exp(repmat((r - div - 0.5*sigma.^2) * T, 1, nMC_max) + sqrt(T) * CSEU)';
XnF_B  = S0 .* exp(repmat((r - div - 0.5*sigma.^2) * T, 1, nMC_max) - sqrt(T) * CSEU)';
Price_EU_A = mean(payoff_fun(XnF_A));
Price_EU_B = mean(payoff_fun(XnF_B));
Price_EU   = exp(-r * T) * (Price_EU_A + Price_EU_B) * 0.5;  % antithetic mean
EU_TOL     = tol * Price_EU;         % absolute tolerance (relative to price)

% Pre‑allocate container for European values
V_EU      = zeros(P, N);
V_EU(:,N) = hS(:,N);                 % at maturity V = payoff
Nm1       = N - 1;                   % last index of backward loop

% -------- Parallel backward induction (European part) -------------
% The heavy Monte‑Carlo integrations across
% paths are parallelised over the time index n.
 

 parfor (n = 1:Nm1, my_pool)  
    % === 7.1. Set simulation nodes at time n ===
    tn  = n * dt;                 % current time
    Xn  = XX(:,:,n);              % price matrix (P × D)
    MLn = mean(log(Xn),2);        % mean of logs (for geometric case)

    % === 7.2. Discount factors from t_n to T ===
    Tmtn  = T - tn;               % time‑to‑maturity from t_n
    dfn   = exp(-r * Tmtn);
    dfn05 = 0.5 * dfn;           % factor for antithetic average

    % === 7.3. Branch on option type for efficiency ===
    if(num_type == 1)              % fast closed‑form for geometric put
        % Pre‑compute log‑returns only once per n (antithetic pairs)
        M_aux=repmat((r - div - 0.5*sigma.^2) * Tmtn,1,nMC_max);
        MLnp_A = mean(M_aux + sqrt(Tmtn) * CSEU, 1)';
        MLnp_B = mean(M_aux - sqrt(Tmtn) * CSEU, 1)';
        MLnp0_A = MLnp_A(1:nMC0);
        MLnp0_B = MLnp_B(1:nMC0);

        for p = 1:P
            MLnp = MLn(p);                % shift for each path p
            MLnF_A = MLnp + MLnp0_A;      % antithetic forward logs (fast)
            MLnF_B = MLnp + MLnp0_B;
            payout = max(K - exp(MLnF_A), 0) + max(K - exp(MLnF_B), 0);

            % --- adaptive MC variance control ---
            std_pay = 0.5 * std(payout);
            cia     = ci_fac * std_pay * dfn / sqrt(nMC0);
            if(cia > EU_TOL)
                nMC = min(round((ci_fac * std_pay * dfn / EU_TOL)^2), nMC_max);
                MLnF_A = MLnp + MLnp_A(1:nMC);
                MLnF_B = MLnp + MLnp_B(1:nMC);
                payout = max(K - exp(MLnF_A), 0) + max(K - exp(MLnF_B), 0);
            end
            V_EU(p,n) = dfn05 * mean(payout);
        end

    else                           % generic pay‑offs (slower pathwise)
        Fnp_A  = exp(repmat((r - div - 0.5*sigma.^2) * Tmtn,1,nMC_max) + sqrt(Tmtn) * CSEU)';
        Fnp_B  = exp(repmat((r - div - 0.5*sigma.^2) * Tmtn,1,nMC_max) - sqrt(Tmtn) * CSEU)';
        Fnp0_A = Fnp_A(1:nMC0,:);
        Fnp0_B = Fnp_B(1:nMC0,:);

        for p = 1:P
            Xnp    = Xn(p,:);           % row vector (1 × D)
            XnF_A  = Xnp .* Fnp0_A;     % forward paths (antithetic)
            XnF_B  = Xnp .* Fnp0_B;
            payout = payoff_fun(XnF_A) + payoff_fun(XnF_B);

            % --- adaptive MC variance control ---
            std_pay = 0.5 * std(payout);
            cia     = ci_fac * std_pay * dfn / sqrt(nMC0);
            if(cia > EU_TOL)
                nMC   = min(round((ci_fac * std_pay * dfn / EU_TOL)^2), nMC_max);
                XnF_A = Xnp .* Fnp_A(1:nMC,:);
                XnF_B = Xnp .* Fnp_B(1:nMC,:);
                payout = payoff_fun(XnF_A) + payoff_fun(XnF_B);
            end
            V_EU(p,n) = dfn * 0.5 * mean(payout);
        end

    end % end branch on num_type
end % end parfor on n

eu_toc = toc(eu_tic);
fprintf("done! (%.0f) ", eu_toc);
clear CSEU Fnp_A Fnp_B Fnp0_A Fnp0_B MLnp_A MLnp_B MLnp0_A MLnp0_B;

%% ---------------------------------------------------------------------
%% 8. INITIALISE AMERICAN COMPONENT ARRAYS
%% ---------------------------------------------------------------------
CV_risk_free = zeros(P,1);      % control value under risk‑free measure
CV_risk_adj  = CV_risk_free;    % control value under risky measure

%% ---------------------------------------------------------------------
%% 9. TERMINAL CONDITIONS AT MATURITY (n = N)
%% ---------------------------------------------------------------------
% At maturity the value equals the intrinsic value.  Both M and V coincide.

n  = N;
MV = hS(:,n);             % M = V = Vh = intrinsic value at maturity

gM = cp * max(0,MV) + cm * min(0,MV);      % collateral cost of exposure
CV_risk_adj  = -dtm * gM;      % initial target for risky GPR (Equation 3.8)

% Train initial GPR models (exact fitting for small design sets)
model_risk_adj  = compact(fitrgp(G(:,:,n), CV_risk_adj , ...
    'BasisFunction', 'none', 'KernelFunction', 'squaredexponential', ...
    'Standardize', false, 'FitMethod', 'exact'));
% Store kernel parameters for warm‑starts in inner loop
kparams_risk_adj  = model_risk_adj .KernelInformation.KernelParameters;
sigma_risk_adj    = model_risk_adj .Sigma;
 
%% ---------------------------------------------------------------------
%% 10. BACKWARD RECURSION FOR AMERICAN OPTION (n = N‑1 ... 0)
%% ---------------------------------------------------------------------
for n = (N-1):-1:0
    % === 10.1. Prepare state variables for step n ===
    if(n > 0)
        Gn    = G(:,:,n);         % Brownian increment at t_n
        Gnp1  = G(:,:,n+1);       % increment at t_{n+1}
        hSn   = hS(:,n);          % intrinsic value at t_n
        V_EUn = V_EU(:,n);        % European continuation value at t_n
        Pmax  = P;                % number of regression points is full P
    else
        % Special case n = 0: use pre‑computed Halton seed (G0) and scalar P=1
        Gn    = G0;
        Gnp1  = G(:,:,1);
        Pmax  = 1;                % only the current state S0
        V_EUn = Price_EU;
        hSn   = EV0;
    end

    % === 10.2. Risk‑free value (CV) ===
    if(n > N-2)   % first backward step: CV = V_EU directly
        CV = V_EUn;
        V_rf  = max(hSn, CV);        % American condition (early exercise)
    else
        % Retrieve kernel parameters from previous risk‑free model
        Alpha   = model_risk_free.Alpha;
        sigmaL  = model_risk_free.KernelInformation.KernelParameters(1);
        sigmaF  = model_risk_free.KernelInformation.KernelParameters(2);
        sigmaL2 = sigmaL^2;  sigmaF2 = sigmaF^2;

        % Analytical integration of SE‑kernel w.r.t. Brownian motion
        R     = chol(CovMat*(dt) + sigmaL2 * eye(D));
        detR  = prod(diag(R)/ sigmaL);
        sigma_aux = df * sigmaF2 / detR;

        % Vectorised computation of the kernel expectation
        parfor (p = 1:Pmax, my_pool)
            Delta_G = (Gn(p,:) - Gnp1)';
            CV_risk_free(p) = exp(-0.5 * sum(Delta_G.*(R\(R'\Delta_G)))) * Alpha;
        end
        CV_risk_free = sigma_aux * max(0, CV_risk_free);
        CV       = CV_risk_free + V_EUn;
        V_rf     = max(hSn, CV);  % apply early‑exercise constraint
    end

    % === 10.3. Risky continuation (CV_h) needed for hedging price ===
    Alpha   = model_risk_adj .Alpha;
    sigmaL  = model_risk_adj .KernelInformation.KernelParameters(1);
    sigmaF  = model_risk_adj .KernelInformation.KernelParameters(2);
    sigmaL2 = sigmaL^2;  sigmaF2 = sigmaF^2;

    R     = chol(CovMat*(dt) + sigmaL2 * eye(D));
    detR  = prod(diag(R)/ sigmaL);
    sigma_aux = df0 * sigmaF2 / detR;

    for p = 1:Pmax
        Delta_G = (Gn(p,:) - Gnp1)';
        aux = exp(-0.5 * sum(Delta_G.*(R\(R'\Delta_G)))) * Alpha;
        CV_risk_adj (p) = aux * sigma_aux;
    end

    % Hedging‑measure continuation (Equation 3.11)
    CV_h = -CV_risk_adj  + df0 * cf * CV;

    % === 10.4. Collateral convention: M = V   or   M = \hat V ===
    if(MVhat)   % Case  M = \hat V  (variant with non‑linear denom.)
        V_ra = (hSn <= 0) .* ((CV_h >= 0) .* (CV_h * dp) + ...
                            (CV_h < 0)  .* max(hSn, CV_h * dm)) + ...
             (hSn > 0)  .* max(hSn, CV_h * dp);
        MV = V_ra;        % mark‑to‑collateral equals V̂
    else          % Case  M = V  (linear Add‑On scheme)
        gn  = cp * max(V_rf,0) + cm * min(0,V_rf);
        V_ra  = max(hSn, CV_h + gn * dtm);
        MV  = V_rf;         % mark‑to‑collateral equals V
    end

    %% === 10.5. Train new GPR models for step n (if n > 0) ===
    if(n > 0)
        % Targets for the two GP regressions at step n
        CV_risk_free   = max(0, V_rf - V_EUn);      % risk‑free excess over EU
        CV_AM_Vh   = max(0, V_rf - V_ra);         % difference V − Vh
        gM        = cp * max(0, MV) + cm * min(0, MV);
        CV_risk_adj    = CV_AM_Vh - dtm * gM;     % Equation 3.13

        % -- Fit risk‑free GP model (warm‑start except first use) --
        if(n > N-2)  % exact fit for the first inner step (cheap)
            model_risk_free = compact(fitrgp(Gn, CV_risk_free, 'Standardize',false, ...
                'KernelFunction','squaredexponential','BasisFunction','none', ...
                'FitMethod','exact'));
        else         % warm‑start with previous hyper‑parameters
            model_risk_free = compact(fitrgp(Gn, CV_risk_free, 'Standardize',false, ...
                'KernelFunction','squaredexponential','BasisFunction','none', ...
                'KernelParameters', kparams_risk_free, 'Sigma', sigma_risk_free, ...
                'FitMethod','exact'));
        end

        % -- Fit risky GP model (always warm‑start)
        model_risk_adj  = compact(fitrgp(Gn, CV_risk_adj , 'Standardize',false, ...
            'KernelFunction','squaredexponential','BasisFunction','none', ...
            'KernelParameters', kparams_risk_adj , 'Sigma', sigma_risk_adj , ...
            'FitMethod','exact')); 
        % Store hyper‑parameters for the next iteration (warm‑start)
        kparams_risk_free = model_risk_free.KernelInformation.KernelParameters;
        sigma_risk_free   = model_risk_free.Sigma;
        kparams_risk_adj  = model_risk_adj .KernelInformation.KernelParameters;
        sigma_risk_adj    = model_risk_adj .Sigma; 
    end
end % end backward loop

%% ---------------------------------------------------------------------
%% 11. COLLECT RESULTS AND RETURN
%% ---------------------------------------------------------------------
Price_rf = V_rf(1);       % American price at t = 0
Price_ra  = V_ra(1);      % Hedged price at t = 0
XVA      = Price_rf - Price_ra;  % total adjustment

end  % ========================= END OF FUNCTION ==========================