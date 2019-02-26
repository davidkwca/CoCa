estimate_confounding_via_kernel_smoothing=function(X,Y){
  # input: (X,Y) where X is an n x d matrix for the potential causes. Here d is the number of variables X_1,...,X_d and n is the number of samples
  #                    Y is a column vector of n instances of the target variable
  
  # output: 2-dimensional vector "parameters" with the confounding parameters (beta,eta) as in the paper, with beta being the relevant one since the 
  # estimation of eta performs rather bad for reasons that I don't completely understand yet. 
  # copyright: Dominik Janzing, 2017
  
  d <<- length(X[1,])
  cxx=cov(X,use='pairwise.complete.obs')
  cxy=cov(X,Y,use='pairwise.complete.obs')
  a_hat=solve(cxx) %*% cxy
  spectral_dec=eigen(cxx)
  spectrumX <<- spectral_dec$values
  normed_spectrum = spectrumX/(max(spectrumX)-min(spectrumX))
  eigenvectors=spectral_dec$vectors
  weights=(t(a_hat) %*% eigenvectors)^2
  weights=weights/sum(weights)
  smoothing_matrix <<- outer(normed_spectrum,normed_spectrum, FUN=Vectorize(kernel))
  smoothed_weights <<- smoothing_matrix %*% t(weights)
  weights_causal <<- rep(1/d,d)
  parameters=optim(c(0,0),distance,method="L-BFGS-B",lower=c(0,0),upper=c(1,10))$par
  return(parameters)
}

distance=function(lambda){
  g=rep(1/sqrt(d),d)
  T=diag(spectrumX)+ lambda[2] * g %*% t(g)
  weights_confounded=spectrumX^{-2} * (g %*% eigen(T)$vectors)^2  
  weights_confounded=weights_confounded / sum(weights_confounded)
  weights_ideal=(1-lambda[1])*weights_causal + lambda[1] * weights_confounded
  smoothed_weights_ideal = smoothing_matrix %*%  t(weights_ideal)
  dist=sum(abs(smoothed_weights - smoothed_weights_ideal))
  return(dist)
}

kernel = function(value1,value2){
  sigma = 0.2
  exp(-(value1-value2)^2 / (2*sigma^2))
}
