function [XVA_MeqV,XVA_MeqVH,Price_RF,Price_MeqV,Price_MeqVH]=XVA_Tree_BK(par)
% XVA_Tree_BK  ────────────────────────────────────────────────────────────
%  CRR‑binomial‑tree implementation of XVA (CVA/DVA/FVA) for a European or
%  Bermudan option under a Black–Karasinski short‑rate framework.
%  The routine propagates three valuations on the same recombining tree:
%     • Price_RF     : risk‑free benchmark (no counterparty / funding risk)
%     • Price_MeqV   : risky price with collateral convention  M =  V
%     • Price_MeqVH  : risky price with collateral convention  M = V̂
%  The corresponding total adjustments are  XVA_MeqV  and  XVA_MeqVH.
%  The algorithm proceeds in three stages for both European and Bermudan
%  exercise styles:   1) risk‑free backward induction;  2) risk adjustment
%  with linear collateral (M = V);  3) risk adjustment with the non‑linear
%  denominator (M = V̂). Comments added 15‑May‑2025.
% ------------------------------------------------------------------------
%  INPUT  (all fields contained in structure par)
%   S0       (1×D)  spot vector of the underlying assets
%   K        (1×1)  strike price (scalar)
%   r        (1×1)  risk‑free short rate (flat)
%   div      (1×D)  dividend rates / funding spreads of each asset
%   CovMat   (D×D)  instantaneous covariance matrix of asset returns
%   T        (1×1)  maturity (years)
%   LB,LC    lender borrowing / collateral spreads
%   RB,RC    receiver borrowing / collateral spreads
%   sF       scalar funding spread
%   N        number of exercise dates for Bermudan option (NE in paper)
%   N_CRR    number of CRR time steps in the binomial tree (NT in code)
%   Type     'CALL_GEO' or 'PUT_GEO'
% ------------------------------------------------------------------------
%  OUTPUT
%   XVA_MeqV   = Price_RF − Price_MeqV
%   XVA_MeqVH  = Price_RF − Price_MeqVH
%   Price_RF   risk‑free option price
%   Price_MeqV option price with M = V collateral convention
%   Price_MeqVH option price with M = V̂ (non‑linear) convention
% ------------------------------------------------------------------------

%% ===[ UNPACK INPUT PARAMETERS ]=========================================
S0      = par.S0;      % spot vector
K       = par.K;       % strike
r       = par.r;       % risk‑free rate
div     = par.div;     % dividend / funding rates
CovMat  = par.CovMat;  % covariance matrix Σ
T       = par.T;       % maturity
LB      = par.LB;      % lender borrowing spread
LC      = par.LC;      % lender collateral spread
RB      = par.RB;      % receiver borrowing spread
RC      = par.RC;      % receiver collateral spread
sF      = par.sF;      % funding spread
NE      = par.N;       % number of Bermudan exercise opportunities
NT      = par.N_CRR;   % binomial steps (CRR tree)

D = length(S0);        % dimension (# of underlyings)

%% ===[ COMPUTE EQUIVALENT VOLATILITY & DIVIDEND ]=======================
%  Map the D‑dimensional basket into a 1‑dim geometric Brownian motion with
%  equivalent volatility σ  and dividend δ so that   S̄_t = geomean(S_t).

t_cor = 0;                                    % cross‑covariance sum
for i = 1:D
    for j = (i+1):D
        t_cor = t_cor + 2*CovMat(i,j);
    end
end

sigma2 = sum(diag(CovMat));                   % diagonal part Σ_ii
sigma  = sqrt(sigma2 + t_cor) / D;            % basket vol (eq.‑dist.)
% Adjusted dividend that keeps drift of geometric average unchanged
div = mean(div) + sigma2*(D-1)/(2*D^2) - t_cor/(2*D^2);

%% ===[ MARKET & TREE PARAMETERS ]=======================================
r0  = r + LB + LC;            % funding‑adjusted short rate

dt  = T/NT;                   % time step (years)
dtm = dt*0.5;                 % half‑step for θ‑scheme in collateral

df  = exp(-r * dt);           % risk‑free discount per step
 df0 = exp(-r0* dt);          % risky  discount per step

u = exp( sigma * sqrt(dt) );  % CRR upward jump factor
 d = 1/u;                      %      downward jump
p = ( exp((r-div)*dt) - d )/(u-d);  % risk‑neutral up‑probability
q = 1-p;                       % down‑probability

% Collateral parameters (positive / negative exposure)
cp = LB + LC*RC - sF;          % coefficient when V ≥ 0
cm = LC + LB*RB;               % coefficient when V ≤ 0

dp = 1/(1 - dtm*cp);          % 1 / (1‑θ cp)  for M = V̂,  pos branch
dm = 1/(1 - dtm*cm);          % idem, neg branch

DNB = NT / NE;                % steps between Bermudan exercise dates

%% ===[ INITIALISE BINOMIAL ARRAYS ]=====================================
Np1 = NT + 1;                 % matrix dimension (max nodes per column)
S  = zeros(Np1, Np1);         % underlying price tree
V  = zeros(Np1, Np1);         % value arrays (duplicated for each stage)

% Build underlying price lattice (normalised, then scaled by geo‑mean)
S(1,1) = 1;                   % root node at t = 0
for n = Np1:-1:1
    for j = 1:n
        S(j,n) = u^(2*j - 1 - n);   % CRR formula S_{j,n}
    end
end
S = S * geomean(S0);         % scale to actual basket spot

%% ===[ TERMINAL PAYOFF ]================================================
if strcmp(par.Type,'CALL_GEO')
    P = max(S - K, 0);
elseif strcmp(par.Type,'PUT_GEO')
    P = max(K - S, 0);
else
    error('XVA_Tree_BK:InvalidType', ...
          'Error: select par.Type as ''CALL_GEO'' or ''PUT_GEO''.');
end
V(:,end) = P(:,end);          % initialise value at maturity

%% ---------------------------------------------------------------------
%% 1) RISK‑FREE BACKWARD INDUCTION (Price_RF) ---------------------------
%% ---------------------------------------------------------------------
for n = NT:-1:1
    np1 = n+1;                              % t_{n+1} column index
    % Vectorised CRR expectation with early‑exercise for Bermudan
    V(1:n,n) = max(0, df * (p * V(2:np1,np1) + q * V(1:n,np1)) );
    if mod(n, DNB) == 0                    % Bermudan exercise date?
        V(:,n) = max(P(:,n), V(:,n));      % allow exercise
    end
end
Price_RF = V(1,1);

%% ---------------------------------------------------------------------
%% 2) RISKY VALUATION ─ Variant  M = V  (linear collateral) -------------
%% ---------------------------------------------------------------------
Vh = zeros(size(V));          % fresh array for V̂ (here M = V)
Vh(:,end) = P(:,end);
for n = NT:-1:1
    np1 = n+1;
    for j = 1:n                     % node‑wise because of gn term
        jp1 = j+1;
        % --- continuation values under risky measure -----------------
        gup = cp*max(V(jp1,np1),0) + cm*min(V(jp1,np1),0);
        vup = dtm*gup + Vh(jp1,np1);

        gdw = cp*max(V(j,np1),0)  + cm*min(V(j,np1),0);
        vdw = dtm*gdw + Vh(j,np1);

        gn  = cp*max(V(j,n),0)    + cm*min(V(j,n),0);
        cv  = df0 * (p*vup + q*vdw) + dtm*gn;  % Eq. (3.9)
        Vh(j,n) = cv;
    end
    if mod(n, DNB) == 0           % Bermudan exercise date?
        Vh(:,n) = max(P(:,n), Vh(:,n));
    end
end
Price_MeqV = Vh(1,1);
XVA_MeqV   = Price_RF - Price_MeqV;

%% ---------------------------------------------------------------------
%% 3) RISKY VALUATION ─ Variant  M = V̂  (non‑linear denominator) -------
%% ---------------------------------------------------------------------
Vh = zeros(size(V));          % reset V̂ array
Vh(:,end) = P(:,end);
for n = NT:-1:1
    np1 = n+1;
    for j = 1:n
        jp1 = j+1;
        % --- continuation values with V̂ on both branches -------------
        gup = cp*max(Vh(jp1,np1),0) + cm*min(Vh(jp1,np1),0);
        vup = dtm*gup + Vh(jp1,np1);

        gdw = cp*max(Vh(j,np1),0)  + cm*min(Vh(j,np1),0);
        vdw = dtm*gdw + Vh(j,np1);

        star = df0 * (p*vup + q*vdw);      % preliminary value "*"
        % Piecewise definition (Eq. 3.14): choose dp/dm by sign criteria
        if star > 0 || P(j,n) > 0
            Vh(j,n) = star * dp;           % positive branch
        else
            Vh(j,n) = star * dm;           % negative branch
        end
    end
    if mod(n, DNB) == 0           % Bermudan exercise date?
        Vh(:,n) = max(P(:,n), Vh(:,n));
    end
end
Price_MeqVH = Vh(1,1);
XVA_MeqVH   = Price_RF - Price_MeqVH;

end  % ============================= END OF FUNCTION ======================
