function [polymodel,win] = polyfitnH2(indepvar,depvar,modelterms)

%--------------------------------------------------------------------------
% "Enhancing Valuation of Variable Annuities in Lévy Models
%  with Stochastic Interest Rate"
% L. Goudenège, A. Molent, X. Wei, A. Zanette
%
% Monte Carlo pricer
% This function compute the polynomial regression and the conditioning
% numebr for the matrix involved in the computation.
% Based on the code by John D'Errico
%
% Author of the code: A. Molent (andrea.molent@uniud.it)
% Date of release: 04 April 2024
%--------------------------------------------------------------------------
 

if nargin<1
  help polyfitn
  return
end

% get sizes, test for consistency
[n,p] = size(indepvar);
if n == 1
  indepvar = indepvar';
  [n,p] = size(indepvar);
end
[m,q] = size(depvar);
if m == 1
  depvar = depvar';
  [m,q] = size(depvar);
end
% only 1 dependent variable allowed at a time
if q~=1
  error 'Only 1 dependent variable allowed at a time.'
end

if n~=m
  error 'indepvar and depvar are of inconsistent sizes.'
end

% check for and remove nans in data
nandata = isnan(depvar) | any(isnan(indepvar),2);
if any(nandata)
  depvar(nandata,:) = [];
  indepvar(nandata,:) = [];
  n = size(indepvar,1);
end

% Automatically scale the independent variables to unit variance
stdind = sqrt(diag(cov(indepvar)));
if any(stdind==0)
  warning 'Constant terms in the model must be entered using modelterms'
  stdind(stdind==0) = 1;
end
% scaled variables
indepvar_s = indepvar*diag(1./stdind);

% do we need to parse a supplied model?
if iscell(modelterms) || ischar(modelterms)
  [modelterms,varlist] = parsemodel(modelterms,p);
  if size(modelterms,2) < p
    modelterms = [modelterms, zeros(size(modelterms,1),p - size(modelterms,2))];
  end  
elseif isscalar(modelterms)
  % do we need to generate a set of modelterms?
  [modelterms,varlist] = buildcompletemodel(modelterms,p);
elseif size(modelterms,2) ~= p
  error 'ModelTerms must be a scalar or have the same # of columns as indepvar'
else
  varlist = repmat({''},1,p);
end
nt = size(modelterms,1);

% check for replicate terms 
if nt>1
  mtu = unique(modelterms,'rows');
  if size(mtu,1)<nt
    warning 'Replicate terms identified in the model.'
  end
end

% build the design matrix
M = ones(n,nt);
scalefact = ones(1,nt);
for i = 1:nt
  for j = 1:p
    M(:,i) = M(:,i).*indepvar_s(:,j).^modelterms(i,j);
    scalefact(i) = scalefact(i)/(stdind(j)^modelterms(i,j));
  end
end

% estimate the model using QR. do it this way to provide a
% covariance matrix when all done. Use a pivoted QR for
% maximum stability.
[Q,R,E] = qr(M,0);

my_rcond=rcond(R);

if(my_rcond>1e-15)
win=1;

polymodel.ModelTerms = modelterms;
polymodel.Coefficients(E) = R\(Q'*depvar);
yhat = M*polymodel.Coefficients(:);

% recover the scaling
polymodel.Coefficients=polymodel.Coefficients.*scalefact;

% variance of the regression parameters
s = norm(depvar - yhat);
if n > nt
  Rinv = R\eye(nt);
  Var(E) = s^2*sum(Rinv.^2,2)/(n-nt);
  polymodel.ParameterVar = Var.*(scalefact.^2);
  polymodel.ParameterStd = sqrt(polymodel.ParameterVar);
else
  % we cannot form variance or standard error estimates
  % unless there are at least as many data points as
  % parameters to estimate.
  polymodel.ParameterVar = inf(1,nt);
  polymodel.ParameterStd = inf(1,nt);
end

% degrees of freedom
polymodel.DoF = n - nt;

% coefficient/sd ratio for a p-value
t = polymodel.Coefficients./polymodel.ParameterStd;

% twice the upper tail probability from the t distribution,
% as a transformation from an incomplete beta. This provides
% a two-sided test for the corresponding coefficient.
% I could have used tcdf, if I wanted to presume the
% stats toolbox was present. Of course, then regstats is
% an option. In that case, the comparable result would be
% found in:    STATS.tstat.pval
polymodel.p = betainc(polymodel.DoF./(t.^2 + polymodel.DoF),polymodel.DoF/2,1/2);

% R^2
% is there a constant term in the model? If not, then
% we cannot use the standard R^2 computation, as it
% frequently yields negative values for R^2.
if any((M(1,:) ~= 0) & all(diff(M,1,1) == 0,1))
  % we have a constant term in the model, so the
  % traditional R^2 form is acceptable.
  polymodel.R2 = max(0,1 - (s/norm(depvar-mean(depvar)) )^2);
  % compute adjusted R^2, taking into account the number of
  % degrees of freedom
  polymodel.AdjustedR2 = 1 - (1 - polymodel.R2).*((n - 1)./(n - nt));
else
  % no constant term was found in the model
  polymodel.R2 = max(0,1 - (s/norm(depvar))^2);
  % compute adjusted R^2, taking into account the number of
  % degrees of freedom
  polymodel.AdjustedR2 = 1 - (1 - polymodel.R2).*(n./(n - nt));
end

% RMSE
polymodel.RMSE = sqrt(mean((depvar - yhat).^2));

% if a character 'model' was supplied, return the list
% of variables as parsed out
polymodel.VarNames = varlist;
else
    win=0;
    polymodel=0;
end

% ==================================================
% =============== begin subfunctions ===============
% ==================================================
function [modelterms,varlist] = buildcompletemodel(order,p)
% 
% arguments: (input)
%  order - scalar integer, defines the total (maximum) order 
%
%  p     - scalar integer - defines the dimension of the
%          independent variable space
%
% arguments: (output)
%  modelterms - exponent array for the model
%
%  varlist - cell array of character variable names

% build the exponent array recursively
if p == 0
  % terminal case
  modelterms = [];
elseif (order == 0)
  % terminal case
  modelterms = zeros(1,p);
elseif (p==1)
  % terminal case
  modelterms = (order:-1:0)';
else
  % general recursive case
  modelterms = zeros(0,p);
  for k = order:-1:0
    t = buildcompletemodel(order-k,p-1);
    nt = size(t,1);
    modelterms = [modelterms;[repmat(k,nt,1),t]];
  end
end

% create a list of variable names for the variables on the fly
varlist = cell(1,p);
for i = 1:p
  varlist{i} = ['X',num2str(i)];
end


% ==================================================
function [modelterms,varlist] = parsemodel(model,p);
% 
% arguments: (input)
%  model - character string or cell array of strings
%
%  p     - number of independent variables in the model
%
% arguments: (output)
%  modelterms - exponent array for the model

modelterms = zeros(0,p);
if ischar(model)
  model = deblank(model);
end

varlist = {};
while ~isempty(model)
  if iscellstr(model)
    term = model{1};
    model(1) = [];
  else
    [term,model] = strtok(model,' ,');
  end
  
  % We've stripped off a model term. Now parse it.
  
  % Is it the reserved keyword 'constant'?
  if strcmpi(term,'constant')
    modelterms(end+1,:) = 0;
  else
    % pick this term apart
    expon = zeros(1,p);
    while ~isempty(term)
      vn = strtok(term,'*/^. ,');
      k = find(strncmp(vn,varlist,length(vn)));
      if isempty(k)
        % its a variable name we have not yet seen
        
        % is it a legal name?
        nv = length(varlist);
        if ismember(vn(1),'1234567890_')
          error(['Variable is not a valid name: ''',vn,''''])
        elseif nv>=p
          error 'More variables in the model than columns of indepvar'
        end
        
        varlist{nv+1} = vn;
        
        k = nv+1;
      end
      % variable must now be in the list of vars. 
      
      % drop that variable from term
      i = strfind(term,vn);
      term = term((i+length(vn)):end);
      
      % is there an exponent?
      eflag = false;
      if strncmp('^',term,1)
        term(1) = [];
        eflag = true;
      elseif strncmp('.^',term,2)
        term(1:2) = [];
        eflag = true;
      end

      % If there was one, get it
      ev = 1;
      if eflag
        ev = sscanf(term,'%f');
        if isempty(ev)
            error 'Problem with an exponent in parsing the model'
        end
      end
      expon(k) = expon(k) + ev;

      % next monomial subterm?
      k1 = strfind(term,'*');
      if isempty(k1)
        term = '';
      else
        term(k1(1)) = ' ';
      end
      
    end
  
    modelterms(end+1,:) = expon;  
    
  end
  
end

% Once we have compiled the list of variables and
% exponents, we need to sort them in alphabetical order
[varlist,tags] = sort(varlist);
modelterms = modelterms(:,tags);


