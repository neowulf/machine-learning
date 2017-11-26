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
	
m = size(X, 1);
for x_i = 1:m
	for k_i = 1:K
		% printf('Processing: %d\n', x_i);
		diff = (X(x_i,:) - centroids(k_i,:)) .^ 2;
		distance(1, k_i) = sqrt(sum(diff));
	end
	[minimum, index] = min(distance(:));
	idx(x_i) = index;


% =============================================================

end