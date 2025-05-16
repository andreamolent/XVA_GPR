function [XVA, ci_XVA, Price_rf, ci_rf, Price_ra, ci_ra] = XVA_LSMC(par)


% ======================================================================
%  XVA_LSMC.m  (commented version)
% ----------------------------------------------------------------------
%  Least‑Squares Monte‑Carlo (LSMC) engine to compute the Credit Valuation
%  Adjustment (XVA) of exotic derivatives.  Two price estimates are
%  produced:
%      • Price_rf  : risk‑free price obtained with the classical Longstaff–
%                    Schwartz algorithm (American option on the basket).
%      • Price_ra  : price under the hedging–measure (collateral/funding
%                    included) using the two variants
%                       –  M = V            (linear add‑on)
%                       –  M = \hat V       (non‑linear denominator)
%  The total XVA is the difference  XVA = Price_rf − Price_ra.
%
%  INPUT ARGUMENTS (scalars unless stated otherwise)
%  ----------------------------------------------------------------------
%  S0      : 1×D vector of spot prices of the underlying assets
%  K       : strike price
%  r       : risk‑free rate
%  div     : 1×D dividend / funding spread vector
%  sigma   : 1×D vector of volatilities (per asset)
%  CS      : D×D Cholesky factor of the log‑return covariance matrix
%  T       : maturity (years)
%  Type    : string identifying the pay‑off:
%                'PUT_GEO'   geometric put    max(K − G, 0)
%                'PUT_ARI'   arithmetic put   max(K − A, 0)
%                'CALL_MAX'  max‑option       max(max(S) − K, 0)
%                'PTF_SWAP'  long‑short swap  max(mean(S₁) − mean(S₂), K)
%  MVhat   : logical, 1 ⇒ use variant   M = \hat V   (non‑linear),
%                         0 ⇒ use variant   M = V     (linear)
%  LB,LC   : lender borrowing / collateral spreads
%  RB,RC   : receiver borrowing / collateral spreads
%  sF      : funding spread
%  M       : number of Monte‑Carlo paths
%  N       : number of time steps in the LSMC discretisation
%  D       : number of underlying assets / basket dimension
%
%  OUTPUT ARGUMENTS
%  ----------------------------------------------------------------------
%  XVA      : total XVA adjustment  (Price_rf − Price_ra)
%  Price_rf : risk‑free American price (mean over MC paths)
%  ci_rf    : 95% confidence half‑width for risk free price Price_rf
%  Price_ra : hedging‑measure price (variant chosen by MVhat)
%  ci_ra    : 95% confidence half‑width for risk adjusted price Price_ra
%
%  IMPLEMENTATION NOTES
%  ----------------------------------------------------------------------
%  • Basis functions for continuation‑value regression are built via a
%    helper routine  polyfitnH()  (high‑order multivariate polynomial).
%  • The algorithm follows these steps per time‑step n (backwards):
%        1.  Evaluate intrinsic value (exercise payoff).
%        2.  Fit regression of discounted continuation value on state S.
%        3.  Decide on early‑exercise and update running payoff vector.
%        4.  Repeat for the risky measure with the appropriate collateral
%            adjustment (cp, cm, dp, dm).
%  • To keep the original behaviour intact **no functional lines were
%    altered**; only explanatory comments have been added.
% ======================================================================

rng(1); % Set random seed for reproducibility

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
M     = par.MC;       % MC points for computing expected value
N     = par.N;       % Number of time steps
D     = par.D;       % Dimension (underlyings) 

%% ------------------------------------------------------------------
%% 0. HOUSEKEEPING
%% ------------------------------------------------------------------
rng(1);                 % reproducible pseudo‑random sequence
wanna_print = 0;        % quick flag – set to 1 for verbose debug prints
if wanna_print
    fprintf("\tXVA_LSMC\n");
end

%% ------------------------------------------------------------------
%% 1. PRE‑COMPUTE CONSTANTS USED THROUGHOUT THE ALGORITHM
%% ------------------------------------------------------------------
% Time and discount factors -----------------------------------------
r0  = r + LB + LC;          % funding‑adjusted short rate

dt  = T / N;                % time‑step
sdt = sqrt(dt);             % √dt  – used for Brownian increments

dtm = 0.5 * dt;             % half time‑step (theta‑scheme)

df  = exp(-r  * dt);        % risk‑free   discount factor
cf  = exp( r  * dt);        % capitalization factor 1/df   (used later)

df0 = exp(-r0 * dt);        % risky discount factor (LB+LC)

% Collateral coefficients (positive / negative exposure) ------------
cp  = LB + LC * RC - sF;     % coefficient for positive exposure
cm  = LC + LB * RB;          % coefficient for negative exposure

% Denominators for the   M = \hat V   scheme ------------------------
dp  = 1 / (1 - dtm * cp);
dm  = 1 / (1 - dtm * cm);

Np1 = N + 1;                % convenience alias (number of time nodes)

%% ------------------------------------------------------------------
%% 2. MAP PAY‑OFF TYPE TO ID CODE AND HANDLE
%% ------------------------------------------------------------------
if strcmp(Type,'PUT_GEO')
    num_type = 1;  minPo = 0;
elseif strcmp(Type,'PUT_ARI')
    num_type = 2;  minPo = 0;
elseif strcmp(Type,'CALL_MAX')
    num_type = 3;  minPo = 0;
elseif strcmp(Type,'PTF_SWAP')
    num_type = 4;
    D1 = D/2;  D2 = D1 + 1;  % indices for long/short halves
    minPo = K;               % minimum payoff for swap
else
    error('XVA_LSMC:UnknownType', 'Unsupported option type.');
end

%% ------------------------------------------------------------------
%% 3. INITIAL MONTE‑CARLO SIMULATION OF UNDERLYING PATHS
%% ------------------------------------------------------------------
S = zeros(M, D, N+1);            % container for simulated prices
S(:,:,1) = repmat(S0, M, 1);     % initial condition S(0) = S0

drift = (r - 0.5*sigma.^2 - div) * dt;   % deterministic part per asset

for n = 2:Np1                         % forward simulation n = 1 … N
    dW = sdt * randn(D, M);            % Brownian increments  ~ N(0,dt)
    F  = CS * dW;                      % correlate increments (D×M)
    S(:,:,n) = S(:,:,n-1) .* exp((F + repmat(drift,1,M))');
end

%% ------------------------------------------------------------------
%% 4. PAY‑OFF FUNCTION HANDLE (vectorised over rows of X)
%% ------------------------------------------------------------------
if num_type == 1      % geometric put
    EV_fun = @(x_) max(K - geomean(x_,2), 0);
elseif num_type == 2  % arithmetic put
    EV_fun = @(x_) max(K - mean(x_,2), 0);
elseif num_type == 3  % max‑call
    EV_fun = @(x_) max(max(x_,[],2) - K, 0);
else                  % portfolio swap
    EV_fun = @(x_) max(mean(x_(:,1:D1),2) - mean(x_(:,D2:end),2), K);
end

%% ------------------------------------------------------------------
%% 5. INITIALISE PAYOFF VECTORS AT MATURITY  (t = T)
%% ------------------------------------------------------------------
PO_rf = EV_fun(S(:,:,Np1));   % risk‑free payoff at maturity
PO_ra = PO_rf;                % start ky‑variant with same terminal payoff

%% ------------------------------------------------------------------
%% 6. BACKWARD LSMC RECURSION  (n = N … 1)
%% ------------------------------------------------------------------
for n = N:-1:1
    %% 6.1.  Intrinsic value & early exercise threshold --------------
    EV  = EV_fun(S(:,:,n));                 % intrinsic value at time n
    min_Pon = minPo * exp(-r * dt + (N+1 - n));  % lower bound (optional)

    %% ------------------- RISK‑FREE OPTION --------------------------
    PO_rf_fut = PO_rf;          % snapshot of future payoff (needed later)
    PO_rf = PO_rf * df;         % discount future payoff to t_n

    IEV = (EV > min_Pon);       % in‑the‑money indicator for regression
    IEV(1:10) = 1;              % ensure at least 10 points (numerical)

    % --- 6.1.a  Regression of continuation value --------------------
    [poly_rf,~] = polyfitnH(S(IEV,:,n), PO_rf(IEV));
    CV_rf = polyvaln(poly_rf, S(:,:,n));   % estimated continuation value

    CV_rf = max(CV_rf, min_Pon + 1e-8);    % enforce positivity threshold

    EN_rf = EV > CV_rf;        % exercise‑now decision (logical vector)
    PO_rf = EV .* EN_rf + (~EN_rf) .* PO_rf; % update payoff vector

    %% ------------------- RISKY / COLLATERALISED OPTION -------------
    % Two distinct formulas depending on variant MVhat (M = V̂ or M = V)
    if MVhat
        % -------- Variant  M = V̂  (non‑linear denominator) ----------
        Mp_fut = max(0, PO_ra);                 % positive exposure in t_{n+1}
        Mm_fut = min(0, PO_ra);                 % negative exposure in t_{n+1}
        g_fut  = cp*Mp_fut + cm*Mm_fut;          % funding cost at t_{n+1}
        en_ra  = df0 * (PO_ra + g_fut*dtm);      % discounted expectation E_n

        % Regression of E_n on current state -------------------------
        [poly_ra,~] = polyfitnH(S(IEV,:,n), en_ra(IEV));
        E = polyvaln(poly_ra, S(:,:,n));         % predicted expectation
        E = max(E, min_Pon + 1e-8);              % numerical floor

        E_pos  = (E > 0);                       % sign of E
        EN_ra_p = EV > E * dp;                  % exercise rule (E > 0)
        EN_ra_m = EV > E * dm;                  % exercise rule (E ≤ 0)

        NEV = (EV < 0);                         % negative intrinsic branch
        if any(NEV)
            % Piecewise update – see Eq. (3.14) in the reference paper
            PO_ra = NEV .* (E_pos .* en_ra * dp + ...
                            (~E_pos) .* (EN_ra_m .* EV + (~EN_ra_m) .* en_ra * dm));
            PO_ra = PO_ra + (~NEV) .* (EN_ra_p .* EV + (~EN_ra_p) .* en_ra * dp);
        else
            PO_ra = EN_ra_p .* EV + (~EN_ra_p) .* en_ra * dp;
        end

    else
        % -------- Variant  M = V   (linear add‑on scheme) ------------
        Mp_fut = max(0, PO_rf_fut);
        Mm_fut = min(0, PO_rf_fut);
        g_fut  = cp*Mp_fut + cm*Mm_fut;           % funding cost future

        Mp_now = max(0, PO_rf);
        Mm_now = min(0, PO_rf);
        g_now  = cp*Mp_now + cm*Mm_now;
        g_now_dtm = g_now * dtm;                  % scaled by dt/2

        en_ra = df0 * (PO_ra + g_fut*dtm);        % discounted expectation
        PO_ra = en_ra + g_now_dtm;                % add current funding cost

        % Regression of continuation value under risky measure -------
        [poly_ra,~] = polyfitnH(S(IEV,:,n), en_ra(IEV));
        E = polyvaln(poly_ra, S(:,:,n));
        CV_ra = E + g_now_dtm;                    % total continuation
        CV_ra = max(CV_ra, min_Pon + 1e-8);

        EN_ra_p = EV > CV_ra;                     % exercise rule
        PO_ra = EV .* EN_ra_p + (~EN_ra_p) .* PO_ra;
    end
end  % ===== end backward loop over n =====

%% ------------------------------------------------------------------
%% 7. PRICE ESTIMATES & 95% CONFIDENCE INTERVALS
%% ------------------------------------------------------------------
Price_rf = mean(PO_rf);
ci_rf    = 1.96 * std(PO_rf) / sqrt(M);

Price_ra = mean(PO_ra);
ci_ra    = 1.96 * std(PO_ra) / sqrt(M);

PO_rf_ra=PO_rf-PO_ra;
XVA = Price_rf - Price_ra;  % total XVA adjustment
ci_XVA    = 1.96 * std(PO_rf_ra) / sqrt(M);

end  % =========================== END OF FUNCTION =====================
