function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% number of training examples
m = size(X, 1);

distToEachCentroid = zeros(m, K); % distToEachCentroid(i, k) is the distance from 
                                  % example X(i) to centroid(k)

for i=1:K
    dist = bsxfun(@minus, X, centroids(i, :)); % m x n
    distToEachCentroid(:, i) = sum(dist .^ 2, 2); % Square sum of each row (m x 1)
end;

% choose closest centroid
[num, idx] = min(distToEachCentroid, [], 2);


% =============================================================

end

