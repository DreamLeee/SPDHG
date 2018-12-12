clear all; close all;

load LVP.mat;
mean_LVP = mean(mean(LVP));
B = LVP - mean_LVP;
B = B';

% mu = 20;
T = 1:1:20;
P = 0.1:0.1:100;

for i = 1:1000
    A(:,i) = sin((2*pi*T)/P(i));
end

x = SparseSPDHG(A,B);
%[x,info] = SparseADMM(A,B);
