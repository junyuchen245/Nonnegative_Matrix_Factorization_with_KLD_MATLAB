% Author: Junyu Chen
% Email: jchen245@jhu.edu
% NMF via KL-Divergence
clear all; close all

d1=112; d2=92; d=d1*d2;
imagesNb=9; peopleNb=40; images=cell(peopleNb,imagesNb);
matX=zeros(d,peopleNb*imagesNb);

jj=1;
for ni=1:peopleNb
    for kimg=1:imagesNb
        filename=strcat('/Users/junyuchen/Documents/MATLAB/MLSP/Lab5/orl_faces/Train/s',num2str(ni),'/',num2str(kimg),'.pgm');
        im=double(imread(filename));
        filename = [];
        
        matX(:,jj)=reshape(im,d,1);
        jj=jj+1;
    end
end


X=matX/max(matX(:));

opt = statset('MaxIter',500,'Display','final');
%[W,H] = nnmf(X,40,'options',opt,'algorithm','mult');
rank = 40;

%[B,W,obj,k,error] = nmf(X,rank,500,0.001); % original NMF
[B,W,obj,k] = ssnmf(X,40,500,0.001,5); % NMF with the sparsity constraint

figure;
for k = 1:rank
    subplot(5,8,k);
    imagesc(reshape(B(:,k),d1,d2));
    colormap gray; axis image off;
end



function [B,W,obj,k,error] = nmf(V, rank, max_iter, lambda)
%%
B = rand(size(V,1),rank);
W = rand(rank,size(V,2));

% W has unit-sum columns (each column should sum to 1)
for i = 1:size(W,2)
    W(:,i) = W(:,i)./sum(W(:,i));
end

% Start iteration
error = [];
old_e = compute_objective(V,B,W);
for iter = 1 : max_iter
    % updates the factorized matrix and the weighting matrix
    disp(['iteration: ',num2str(iter)])
    B = B .* (((V./(B*W))*W')./ sum(W',1));
    W = W .* ((B'*(V./(B*W)))./ sum(B',2));
    obj = compute_objective(V,B,W);
    error = [error obj];
    if(abs(obj - old_e) <= lambda)
        error = [error obj];
        break;
    end
    old_e = obj;
end
k = iter;
end

function [B,W,obj,k,error] = ssnmf(V, rank, max_iter, lambda, alpha)
%%
B = rand(size(V,1),rank);
W = rand(rank,size(V,2));

% W has unit-sum columns (each column should sum to 1)
for i = 1:size(W,2)
    W(:,i) = W(:,i)./sum(W(:,i));
end

% Start iteration
error = [];
old_e = compute_objective_ss(V,B,W,alpha);
for iter = 1 : max_iter
    disp(['iteration: ',num2str(iter)])
    % updates the factorized matrix and the weighting matrix
    B = B .* (((V./(B*W))*W')./ sum(W',1));
    W = W .* ((B'*(V./(B*W)))./ sum(B',2));
    obj = compute_objective_ss(V,B,W,alpha);
    error = [error obj];
    if(abs(obj - old_e) <= lambda)
        error = [error obj];
        break;
    end
    old_e = obj;
end
k = iter;
end

%% KL-Divergence obejctive function
% B: the factorized matrix
% W: weights
function [obj] = compute_objective(V,B,W)
part1 = V.*log(V./(B*W));
part1(isnan(part1))=0;
part1 = sum(part1(:));

obj = part1+sum(V(:))-sum(sum(B*W));
end

%% KL-Divergence imposing sparsity (L-1 minimization)
% Imposing sparsity on the weight matrix W (L-1 norm).
% B: the factorized matrix
% W: weights
function [obj] = compute_objective_ss(V,B,W,alpha)
part1 = V.*log(V./(B*W));
part1(isnan(part1))=0;
part1 = sum(part1(:));
obj = part1+sum(V(:))-sum(sum(B*W)) + alpha*sum(W(:));
end