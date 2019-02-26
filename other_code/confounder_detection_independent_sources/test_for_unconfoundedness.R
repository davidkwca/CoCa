test_for_unconfoundedness = function(X,Y){
  
  #  This is the function used to test unconfoundedness as described in Section 7.2 of the arXiv paper 
  #  'Detecting non-causal artifacts in multivariate linear regression models' 
  #  by Dominik Janzing and Bernhard Schoelkopf, 2018.  
  
  # input: X (n x d matrix with n d-dimensional observations of the potential causes)
  #        Y (n x 1 matrix with n observations of the target quantity)  
  
  # output: p-value for the test for non-confounding
  
  # set number of reference samples the test statistics is compared to
  n_ref =1000
  d = dim(X)[2]
  Cxx = cov(X)
  Conxx <<- solve(Cxx)
  Cxy = cov(X,Y)
  aHat = Conxx %*% Cxy
  #aHatNorm = aHat/sqrt(sum(aHat^2))
  M = matrix(rnorm(n_ref*d,0,1),nrow=d,ncol=n_ref)
  samples = rep(0,n_ref)
  for (j in 1:n_ref){
    samples[j] = test_statistic(M[,j])
  }
  pvalue = length(samples[samples>test_statistic(aHat)])/n_ref
}

test_statistic = function(v){
  d = length(d)
  sum(v * (Conxx %*% v))/sum(v^2) -sum(diag(Conxx))/d
}