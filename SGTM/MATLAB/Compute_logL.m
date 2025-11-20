function [logL,Norm_logL, post] = Compute_logL(GM,ED)

normal = (2*pi)^(size(ED,2)/2);
% Compute Mahalanobis distance between any point in ED and any center of GM
a = exp(-0.5.*mahal_d(ED,GM.mu,GM.Sigma));
DetC = zeros(size(GM.Sigma,3),1);
for k=1:size(GM.Sigma,3)
    DetC(k) = det(GM.Sigma(:,:,k));
end
% Divide by determinant of every component's covariance matrix
act = a./repmat(normal.*sqrt(DetC'),size(a,1),1);




% Multiply the sum of all activations by the prior (1/NumComp)
l1 = log(sum(act,2)./size(a,2));
logL = sum(l1);
Norm_logL = logL./size(ED,1);

post = act.*size(a,2);
s = sum(post, 2);
% Set any zeros to one before dividing 
s = s + (s==0);

k = size(GM.mu);
disp("here")
disp(k)

post = post./(s*ones(1, size(GM.mu,1)));



