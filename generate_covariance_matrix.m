function S = generate_covariance_matrix(d, sigma_min,sigma_max, rho_min, rho_max)
% Genera una matrice di correlazione dxd con valori compresi tra rho_min e rho_max
% Assicura che la matrice sia simmetrica e definita positiva

% Passo 1: Creazione di una matrice simmetrica con valori casuali tra rho_min e rho_max
R = rho_min + (rho_max - rho_min) * rand(d, d);
R = (R + R') / 2; % Simmetrizzazione

% Passo 2: Imposta la diagonale a 1 (proprietà delle matrici di correlazione)
for i = 1:d
    R(i, i) = 1;
end

% Passo 3: Correggere la matrice per renderla definita positiva
[V, D] = eig(R);
D(D < 0) = 0; % Imposta gli autovalori negativi a zero
C = V * D * V'; % Ricostruzione della matrice

% Passo 4: Normalizzazione per garantire una diagonale unitaria
stddev = sqrt(diag(C));
C = C ./ (stddev * stddev');

% Assicura che la matrice sia esattamente simmetrica a causa di eventuali errori numerici
C = (C + C') / 2;

eigss=eig(C);
while(min(eigss)<1e-6)
    C=0.9999*C+0.0001*eye(d);
    eigss=eig(C);
end

sigma = sigma_min + (sigma_max - sigma_min) * rand(d, 1);

% Costruisce la matrice di covarianza
S = diag(sigma) * C * diag(sigma);

% Assicura che la matrice sia esattamente simmetrica a causa di errori numerici
S = (S + S') / 2;
end