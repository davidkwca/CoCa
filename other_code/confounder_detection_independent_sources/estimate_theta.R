estimate_theta = function(Cxx,ahat){
  
  #  This is the function used to estimate the parameter theta  as used in the arXiv paper 
  #  'Detecting non-causal artifacts in multivariate
  #  linear regression models' by Dominik Janzing and Bernhard Schoelkopf, 2018.   
  #  it uses the maximum likelihood estimation described in eq. (10)
   
  # input: Cxx  (covariance matrix of the potential causes X)
  #        ahat (regression vector obtained by regressing Y on X)  
  # output: theta_est (estimated value of theta)
  
  # calls the function 'density.R' computing the probability densioty given by eq. (10)
  
  my_path = '~/bayes/other_code/confounder_detection_independent_sources/'
  source(paste(my_path, 'density.R', sep=''))
  Cxx <<- Cxx
  ahat <<- ahat
  dimension <<- length(ahat)
  theta_est = optim(par=0,loglikeli,method='L-BFGS-B',lower =0,upper=1000)$par
  x = 1:100/100
  y = lapply(x,loglikeli)
  #plot(x,y)
  #readline()
  return(theta_est)
}


loglikeli = function(theta){
  Msquared = diag(dimension) + theta * solve(Cxx)
  M = Msqrt(Msquared)
  return(-1*log(density(M,ahat)))
}

Msqrt = function(matrix){
  spectraldec = eigen(matrix)
  U = spectraldec$vectors
  eigenvalues = spectraldec$values
  output = U %*% diag(sqrt(eigenvalues)) %*% t(U)
}
