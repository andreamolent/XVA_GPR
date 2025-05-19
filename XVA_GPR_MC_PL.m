function [XVA,Price_rf,Price_ra]=XVA_GPR_MC_PL(par,my_pool)
% XVA pricer with GPR‑MC approach
% Code author: Molent Andrea
% Creation: 19 May 2022
% Last update: 14 May 2025
% -------------------------------------------------------------------------
%  INPUT ARGUMENTS
%  ------------------------------------------------------------------------
%  par      : structure with all model, market and contract parameters
%  my_pool  : parallel pool object used in parfor loops (can be [])
%
%  ------------------------------------------------------------------------
%  OUTPUT ARGUMENTS
%  ------------------------------------------------------------------------
%  XVA      : total valuation adjustment
%  Price_rf : American (risk‑free) option price at t = 0
%  Price_ra  : Hedged price (collateralised measure) at t = 0
%
%  ------------------------------------------------------------------------
%  INTERNAL WORK‑FLOW
%  ------------------------------------------------------------------------
%  This function prices the total valuation adjustment (XVA) of exotic
%  derivatives using a Gaussian‑Process Regression combined with Monte‑Carlo
%  (GPR‑MC) technique – see Molent (2022) for details.
%
%  The algorithm proceeds as follows:
%    1.  Unpack input parameters and pre‑compute constants that will be
%        reused throughout the simulation.
%    2.  Map the option type string (e.g. 'PUT_GEO') to a numeric code and
%        create a vectorised payoff function handle.
%    3.  Build a quasi‑random Halton design‑set (P×D) for GPR training.
%    4.  Compute European  values via adaptive, antithetic MC.
%    5.  Perform a backward dynamic‑programming recursion for the American
%        component under two measures: risk‑free and hedging (risky).
%    6.  At each step train / warm‑start two GP models (fitrgp) in parallel.
%    7.  Return the American price (Price_rf), the risk adjusted price 
%        (Price_ra) and the total adjustment XVA = Price_rf − Price_ra.
 

%% ---------------------------------------------------------------------
%% 1. UNPACK INPUT PARAMETERS
%% ---------------------------------------------------------------------
S0    = par.S0;      % Spot vector (1 × D)
K     = par.K;       % Strike
r     = par.r;       % Risk‑free rate
div   = par.div;     % Dividend / funding spread vector (1 × D)
sigma = par.sigma;   % Volatility vector (1 × D)
CS    = par.CS;      % Cholesky factor of Brownian covariance (D × D)
T     = par.T;       % Maturity (years)
LB    = par.LB;      % Lender borrowing spread
LC    = par.LC;      % Lender collateral spread
RB    = par.RB;      % Receiver borrowing spread
RC    = par.RC;      % Receiver collateral spread
sF    = par.sF;      % Funding spread
Type  = par.Type;    % Pay‑off identifier string
MVhat = par.MVhat;   % Flag: 1 ⇒ M = \hat V_rf,  0 ⇒ M = V_rf
P     = par.P;       % Design‑set size for GPR
M     = par.M;       % MC points for computing expected value
N     = par.N;       % Number of time steps
D     = par.D;       % Dimension (underlyings)
cl    = par.cl;      % Confidence level for MC CI
tol   = par.tol;     % Absolute tolerance on MC estimator

%% ---------------------------------------------------------------------
%% 2. PRE‑COMPUTE CONSTANTS (used many times)
%% ---------------------------------------------------------------------
r0    = r + LB + LC;           % Funding‑adjusted rate

dt    = T / N;                 % Time step
% Half‑step used in several collateral formulas
dtm   = dt * 0.5;

df    = exp(-r  * dt);         % Discount factor under risk‑free measure
df0   = exp(-r0 * dt);         % Discount factor under risky measure
cf  = 1/df;                  % Convenience factor (pre‑computed reciprocal)

% Collateral coefficients (positive / negative exposure)
cp    = LB + LC * RC - sF;
cm    = LC + LB * RB;
% Denominators for the non‑linear variant M = \hat V_rf
dp    = 1 / (1 - dtm * cp);
dm    = 1 / (1 - dtm * cm);

% Auxiliary factors reused later (avoid recomputation in loops) 
cp_dtm   = cp * dtm;             % Pre‑multiplied to save flops
cm_dtm   = cm * dtm;

% Kernel & basis choice for fitrgp (tuned to speed and stability)
my_kernelFun = 'matern32';
my_basisFun  = 'none';

%% ---------------------------------------------------------------------
%% 3. MAP OPTION TYPE → CODE & PAYOFF HANDLE (vectorised)
%% ---------------------------------------------------------------------
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
    num_type = 4;
    D1 = D/2; D2 = D1 + 1;      % split basket in two halves
    payoff_fun = @(X_) max(mean(X_(:,1:D1),2) - mean(X_(:,D2:end),2), K);
else
    error('XVA_GPR_MC_PL:UnknownType', ...
          'Unsupported option type "%s".', Type);
end

%% ---------------------------------------------------------------------
%% 4. LOW‑DISCREPANCY GRIDS (Halton) FOR STATE & MC INCREMENTS
%% ---------------------------------------------------------------------
% 4a. Halton grid (P × D) for GPR design‑set -----------------------------
prime_num = primes(1000);
 Halton = haltonset(D,'Skip',1);
 
Halton_points = net(Halton, P-1);
G_design = icdf(makedist('Normal'), Halton_points');   % (D × P-1)

% 4b. Halton grid (M × D) for Brownian increments in MC (antithetic) ----- 
Halton2   = haltonset(D,'Skip',1e3,'Leap',prime_num(D+1)-1); 
G_MC = icdf(makedist('Normal'), net(Halton2, par.M)');

GG   = sqrt(dt) * CS * G_MC;                  % scaled Brownian increments
% Deterministic drift term for each asset (row vector 1 × D)
drift = (r - 0.5*sigma.^2 - div) * dt;
F     = exp((GG + repmat(drift, 1, M))');      % Multiplier matrix (M × D)

%% ---------------------------------------------------------------------
%% 5. PRE‑ALLOCATE MAIN CONTAINERS
%% ---------------------------------------------------------------------
V_rf     = zeros(P,1);           % V_rf  : value under risk‑free measure
V_ra    = V_rf;                    % V̂ : value under hedging measure
XX    = zeros(P,D,N);         % Simulated prices  (P × D × N)
hS    = zeros(P,N);           % Intrinsic values  (P × N)
V_EU  = hS;                   % European values

CSGX  = CS * G_design;        % Pre‑scale for performance
ci_fac = norminv(1 - cl/2);   % Z‑score for two‑sided CI

% Adaptive MC settings ---------------------------------------------------
nMC0    = 1e3;                                 % initial paths per node
nMC_max = max(1e5, 4e7 / D);                  % dimension‑aware upper bound
if(num_type == 4)                              % swap is slower ⇒ fewer
    nMC_max = nMC_max / 4;
end

%% ---------------------------------------------------------------------
%% 6. EUROPEAN COMPONENT (ANTITHETIC MC)
%% --------------------------------------------------------------------- 
% Generate antithetic normals once for the whole EU routine ------------
CSEU   = CS * randn(D, nMC_max);

% --- Terminal payoff (t = T) ------------------------------------------ 
XnF_A = S0 .* exp(repmat((r-div-0.5*sigma.^2)*T,1,nMC_max) + sqrt(T)*CSEU)';
XnF_B = S0 .* exp(repmat((r-div-0.5*sigma.^2)*T,1,nMC_max) - sqrt(T)*CSEU)';
Price_EU_A = mean(payoff_fun(XnF_A));
Price_EU_B = mean(payoff_fun(XnF_B));
Price_EU   = exp(-r*T) * (Price_EU_A + Price_EU_B) * 0.5;  % antithetic avg
EU_TOL     = tol * Price_EU;                               % abs tolerance

V_EU(:,N) = hS(:,N);               % initial condition at maturity
Nm1 = N - 1;                       % last index for backward EU loop

% ----------------------------------------------------------------------
% Parallel loop across time steps n = 1 : N-1 (European part) -----------
% ---------------------------------------------------------------------- 

parfor (n = 1:Nm1, my_pool)   
    %% 6.1. Build design‑set nodes at time n ----------------------------
    tn  = n * dt;
    Wt  = sqrt(tn) * CSGX;                               % (D × P-1)
    Xn  = [S0; S0 .* exp(repmat((r-div-0.5*sigma.^2)*tn,1,P-1) + Wt)'];
    XX(:,:,n) = Xn;                                      % store prices

    %% 6.2. Intrinsic value h(S) ---------------------------------------
    hSn      = payoff_fun(Xn);
    hS(:,n)  = hSn;

    %% 6.3. European via antithetic MC --------------------
    MLn = mean(log(Xn),2);                % only for geometric put accel.
    Tmtn = T - tn;                        % residual time to maturity
    dfn  = exp(-r * Tmtn);
    dfn05= dfn * 0.5;                     % ½ factor for antithetic avg

    if(num_type == 1)  % ---- Fast path for geometric put -------------
        % Pre‑compute antithetic log‑returns (independent of p)
        MLnp_A = mean(repmat((r-div-0.5*sigma.^2)*Tmtn,1,nMC_max) + ...
                      sqrt(Tmtn)*CSEU, 1)';
        MLnp_B = mean(repmat((r-div-0.5*sigma.^2)*Tmtn,1,nMC_max) - ...
                      sqrt(Tmtn)*CSEU, 1)';
        MLnp0_A = MLnp_A(1:nMC0);
        MLnp0_B = MLnp_B(1:nMC0);

        for p = 1:P
            MLnp   = MLn(p);
            MLnF_A = MLnp + MLnp0_A;      % antithetic forward logs
            MLnF_B = MLnp + MLnp0_B;
            payout = max(K - exp(MLnF_A),0) + max(K - exp(MLnF_B),0);

            % --- adaptive variance control ---------------------------
            std_pay = 0.5 * std(payout);
            cia     = ci_fac * std_pay * dfn / sqrt(nMC0);
            if(cia > EU_TOL)
                nMC = min(round((ci_fac * std_pay * dfn / EU_TOL)^2), ...
                           nMC_max);
                MLnF_A = MLnp + MLnp_A(1:nMC);
                MLnF_B = MLnp + MLnp_B(1:nMC);
                payout = max(K - exp(MLnF_A),0) + max(K - exp(MLnF_B),0);
            end
            V_EU(p,n) = dfn05 * mean(payout);
        end

    else              % ---- Generic path (all other pay‑offs) --------
        Fnp_A = exp(repmat((r-div-0.5*sigma.^2)*Tmtn,1,nMC_max) + ...
                    sqrt(Tmtn)*CSEU)';
        Fnp_B = exp(repmat((r-div-0.5*sigma.^2)*Tmtn,1,nMC_max) - ...
                    sqrt(Tmtn)*CSEU)';
        Fnp0_A = Fnp_A(1:nMC0,:);
        Fnp0_B = Fnp_B(1:nMC0,:);

        for p = 1:P
            Xnp   = Xn(p,:);
            XnF_A = Xnp .* Fnp0_A;           % antithetic forward paths
            XnF_B = Xnp .* Fnp0_B;
            payout_A=payoff_fun(XnF_A);
            payout_B=payoff_fun(XnF_B);
            payout   = payout_A + payout_B;

            % --- adaptive variance control ---------------------------
            std_pay = 0.5 * std(payout);
            cia     = ci_fac * std_pay * dfn / sqrt(nMC0);
            if(cia > EU_TOL)
                nMC   = min(round((ci_fac * std_pay * dfn / EU_TOL)^2), ...
                              nMC_max);
                XnF_A = Xnp .* Fnp_A(1:nMC,:);
                XnF_B = Xnp .* Fnp_B(1:nMC,:);
                payout_A=payoff_fun(XnF_A);
                payout_B=payoff_fun(XnF_B);
            end
            V_EU(p,n) = dfn * 0.5 * (mean(payout_A) + mean(payout_B));
        end
    end % end branch on num_type
end % end parfor (European)


%% ---------------------------------------------------------------------
%% 7. AMERICAN COMPONENT – BACKWARD DYNAMIC PROGRAMMING
%% ---------------------------------------------------------------------
% First backward step (n = N-1) is treated explicitly outside loops
n  = N - 1;
Xn = XX(:,:,n);
hSn = hS(:,n);
V_EUn = V_EU(:,n);
V_rf     = max(V_EUn, hSn);          % apply early‑exercise condition

if(MVhat)  % ===== Case M = \hat V_rf  (non‑linear denominator) ========== 
    parfor (p = 1:P, my_pool)
        XF           = Xn(p,:) .* F;        % forward prices matrix (M × D)
        Vh_gM_fut    = payoff_fun(XF);  % intrinsic payoff at t = T
        g_fut        = cp * max(Vh_gM_fut,0) + cm * min(Vh_gM_fut,0);
        E            = df0 * (dtm * mean(g_fut) + mean(Vh_gM_fut));
        if(hSn(p) < 0)
            % piecewise definition (Equation 3.14)
            if(E > 0)
                V_ra(p) = E * dp;
            else
                V_ra(p) = max(hSn(p), E * dm);
            end
        else
            V_ra(p) = max(hSn(p), E * dp);
        end
    end
else        % ===== Case M = V_rf  (linear add‑on) ======================= 
    parfor (p = 1:P, my_pool)
        XF = Xn(p,:) .* F;
        % Compute V̂ future intrinsic value efficiently branch‑wise
        Vh_gM_fut= payoff_fun(XF);
        g_fut = cp_dtm * max(Vh_gM_fut,0) + cm_dtm * min(Vh_gM_fut,0);
        E     = df0 * (mean(g_fut) + mean(Vh_gM_fut));
        g_now = cp_dtm * max(V_rf(p),0) + cm_dtm * min(V_rf(p),0);
        V_ra(p) = E + g_now;                     % Equation 3.9
    end
    V_ra = max(V_ra, hSn);                         % enforce early exercise
end

% Targets for the first GP fit ------------------------------------------------
CV_risk_free = max(0, V_rf - V_EUn);
CV_risk_adj  = max(0, V_rf - V_ra);
if(MVhat), MV = V_ra; else, MV = V_rf; end

gM       = cp * max(MV,0) + cm * min(MV,0);
CV_risk_adj   = CV_risk_adj  - dtm * gM;         

% Fit initial GP models -------------------------------------------------------
model_risk_free = compact(fitrgp(Xn, CV_risk_free, 'BasisFunction', my_basisFun, ...
    'KernelFunction', my_kernelFun, 'Standardize', true));
model_risk_adj  = compact(fitrgp(Xn, CV_risk_adj , 'BasisFunction', my_basisFun, ...
    'KernelFunction', my_kernelFun, 'Standardize', true));

% Store hyper‑parameters for warm‑starts in the main loop
kparams_risk_free = model_risk_free.KernelInformation.KernelParameters;
sigma_risk_free   = model_risk_free.Sigma;

kparams_risk_adj  = model_risk_adj .KernelInformation.KernelParameters;
sigma_risk_adj    = model_risk_adj .Sigma;

const_mrf = parallel.pool.Constant(model_risk_free);  % broadcast for parfor
const_mry = parallel.pool.Constant(model_risk_adj );

%% ---------------------------------------------------------------------
%% 8. MAIN BACKWARD LOOP (n = N-2 : 0)
%% ---------------------------------------------------------------------
for n = (N-2):-1:0
    if(n > 0)
        Pmax  = P;                       % full design‑set size
        Xn    = XX(:,:,n);
        hSn   = hS(:,n);
        V_EUn = V_EU(:,n);
    else
        % Special case n = 0: current state is a single point S0
        Xn    = S0;
        hSn   = hS(1,1);
        Pmax  = 1;
        V_EUn = Price_EU;
    end

    if(MVhat)  % ===== Variant  M = \hat V_rf  ============================
        V_rf  = zeros(Pmax,1);
        V_ra = zeros(Pmax,1);
        parfor (p = 1:Pmax, my_pool)
            %% 8.1. Future states under MC increment -------------------
            XF = Xn(p,:) .* F;                % (M × D)
            %% 8.2. Predict control variate under both measures -----------
            CV_risk_free_fut = max(0, const_mrf.Value.predict(XF));
            CV_risk_adj_fut = const_mry.Value.predict(XF);
            %% 8.3. Risk‑free control value ----------------------
            CV = df * mean(CV_risk_free_fut) + V_EUn(p);
            V_rf(p) = max(hSn(p), CV);            % early exercise
            %% 8.4. Hedging control value ------------------------
            Vh_gM_fut = CV * cf - mean(CV_risk_adj_fut);
            E = df0 * Vh_gM_fut;
            % Piecewise formula (Equation 3.14)
            if hSn(p) <= 0
                V_ra(p) = max(hSn(p), (E > 0) * E * dp + (E <= 0) * E * dm);
            else
                V_ra(p) = max(hSn(p), E * dp);
            end
        end

    else        % ===== Variant  M = V_rf  ================================
        parfor (p = 1:Pmax, my_pool)
            XF = Xn(p,:) .* F;
            CV_risk_free_fut = max(0, const_mrf.Value.predict(XF));
            CV_risk_adj_fut = const_mry.Value.predict(XF);
            CV = df * mean(CV_risk_free_fut) + V_EUn(p);
            V_rf(p) = max(hSn(p), CV);
            Vh_gM_fut = CV * cf - mean(CV_risk_adj_fut);
            E = df0 * Vh_gM_fut;
            g_now = cp * max(V_rf(p),0) + cm * min(V_rf(p),0);
            V_ra(p) = E + dtm * g_now;
        end
        V_ra = max(V_ra, hSn);
    end

    %% 8.5. TRAIN GP MODELS FOR STEP n (unless n = 0) ------------------
    if(n > 0)
        CV_risk_free = max(0, V_rf - V_EUn);
        CV_risk_adj  = max(0, V_rf - V_ra);
        if(MVhat), MV = V_ra; else, MV = V_rf; end
        gM       = cp * max(MV,0) + cm * min(MV,0);
        CV_risk_adj   = CV_risk_adj  - dtm * gM;

        % --- Risk‑free GP (warm‑started) -----------------------------
        try
            model_risk_free = compact(fitrgp(Xn, CV_risk_free, 'BasisFunction', my_basisFun, ...
                'KernelFunction', my_kernelFun, 'Standardize', true, ...
                'KernelParameters', kparams_risk_free, 'Sigma', sigma_risk_free));
        catch
            % fall‑back: fit from scratch if warm‑start fails
            model_risk_free = compact(fitrgp(Xn, CV_risk_free, 'BasisFunction', my_basisFun, ...
                'KernelFunction', my_kernelFun, 'Standardize', true));
        end
        kparams_risk_free = model_risk_free.KernelInformation.KernelParameters;
        sigma_risk_free   = model_risk_free.Sigma;

        % --- Risky GP (warm‑started) ---------------------------------
        try
            model_risk_adj  = compact(fitrgp(Xn, CV_risk_adj , 'BasisFunction', my_basisFun, ...
                'KernelFunction', my_kernelFun, 'Standardize', true, ...
                'KernelParameters', kparams_risk_adj , 'Sigma', sigma_risk_adj ));
        catch
            model_risk_adj  = compact(fitrgp(Xn, CV_risk_adj , 'BasisFunction', my_basisFun, ...
                'KernelFunction', my_kernelFun, 'Standardize', true));
        end
        kparams_risk_adj  = model_risk_adj .KernelInformation.KernelParameters;
        sigma_risk_adj    = model_risk_adj .Sigma;

        % Broadcast new models for next iteration --------------------
        const_mrf = parallel.pool.Constant(model_risk_free);
        const_mry = parallel.pool.Constant(model_risk_adj );
    end
end % end backward loop

%% ---------------------------------------------------------------------
%% 9. OUTPUT RESULTS ----------------------------------------------------
%% ---------------------------------------------------------------------
Price_rf = V_rf(1);        % American price at t = 0
Price_ra  = V_ra(1);       % Hedged price at t = 0
XVA      = Price_rf - Price_ra;   % total valuation adjustment

end % ========================== END OF FUNCTION ==========================
