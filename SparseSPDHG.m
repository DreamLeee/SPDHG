function x_global = SparseSPDHG(A,B)
% Solve the sparse liner regression minimization problem by SPDHG

% minimize the objective function: \sum_{i=1}^N 0.5*||A*x - b_i||+y'*(F*x)
% where y is the max value.

% ---------------------------------------------
% Input:
%       A       -    d*na matrix  d = 20;  na = n;
%       B       -    d*nb matrix  nb = 1;  column: sampla, row: time;
%       F       -    the sparse matrix, F = eye(size(n));
%       beta    -    the parameter of step;
%       L       -    the parameter of constant in beta;
%       gamma   -    the parameter in L;
%       s       -    the slack variable of prime-dual;
%
% Output:
%       X       -    na*nb matrix
%       info    -    dual variable 
%             iter      -    number of iterations                  
%             Valobj    -    sum_{i=1}^N 0.5*||A*x - b_i||+ y'*(F*x)
%
% ---------------------------------------------
% Version 1.0 - 11/12/2018, written by Meng Lee.
%
% 
%Current problem: How to set the initialization.
%


maxiter = 10000;
n_basefunction = size(A,2);

I = eye(n_basefunction);
s = 1;
gamma = 1;
L = 0;

x = ones(n_basefunction,1);
y = zeros(n_basefunction,1);
J_history = zeros(maxiter,1);

ATA = A'*A;


for i = 1:maxiter
    
    b_idx = randperm(size(B,2),1);
    b = B(:,b_idx);
    
    % update x
    x_old = x;
    L = max(8*gamma*max(eig(ATA)),sqrt(8*L^2 + gamma*max(eig(ATA))));
    beta = 1/(sqrt(i+1)+L);
    x = x_old - beta * (gradient(obj_PD(A,x,b,0)) + I * y);
%   x = (1/(t+1))*x;
    
    % update y
    y_old = y;
    ObjectiveFunction_max = @(z)(-1)*(0.5*((A*x_old-b)'*(A*x_old-b)) + z'*x_old);
    [~,y_tr] = fmincon(ObjectiveFunction_max,A(1,:)',[],[],[],[],[],[]);
    
    ObjectiveFunction_min = @(z)(-y_tr - (1/(2*s))*((z-y_old)'*(z-y_old)));
    [y,~] = fmincon(ObjectiveFunction_min,A(1,:)',[],[],[],[],[],[]);
    
    y = -y;        
    J_history(i) = obj_PD(A,x,B,y);
    
end
    x_global = x;
end


function J = obj_PD(A,x,B,y)
% the objective function: \sum_{i=1}^N 0.5*||A*x - b_i||^2+ y'*(F*x)
J = 0;
predications = A*x;

for i = 1:size(B,2)
Errors = predications - B(:,i);
J = J + 0.5*(Errors'*Errors);
end

J = J + y'*(eye(length(x))*x);
end



