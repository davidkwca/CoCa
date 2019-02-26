simulate_data_and_test_confounding = function(d,l,n,runs){
  
 #  This is the function producing the plot in Figure 4 of the arXiv paper 
 #  'Detecting non-causal artifacts in multivariate
 #  linear regression models' by Dominik Janzing and Bernhard Schoelkopf, 2018.   
 #  inputs: d (dimension)
 #          l (number of sources) 
 #          n (sample size)
 #  output: plot true confounding beta vs p-value for 'runs' many instances 
 #          
 #  copyright: Dominik Janzing, 2018.   
  
 # uses 'test_for_unconfoundedness.R'  
 
 source('~/nonlinngam/code/confounder_detection_independent_sources/test_for_unconfoundedness.R')
 
 library(MASS)  
  
  theta = rep(0,runs)
  theta_est = rep(0,runs)
  beta = rep(0,runs)
  beta_est = rep(0,runs)
  pvalue = rep(0,runs)
  loglike = rep(0,runs)
  for (j in 1:runs){
     cat('j = ',j,'\n')  
     # generate sources
     S = matrix(rnorm(n*l,0,1),nrow=l,ncol=n)
     # generate mixing matrix
     M = matrix(rnorm(d*l,0,1),nrow=d,ncol=l)
     # generate X
     X = M %*% S
     # generate sigma_a and sigma_c
     sigma_a =  runif(1,0,1)
     sigma_c =  runif(1,0,1)/sqrt(sum(diag(solve(M %*% t(M))))/d)
     # generate structure coefficients
     a = rnorm(d,0,sigma_a)
     # generate confounding vector
     c = rnorm(l,0,sigma_c)
     # generate Y
     Y = a %*% X + c %*% S
     
     # compute confounding strength. First compute the perturbation of the regression vector a caused by confounding
     aPert = t(ginv(M)) %*% c
     beta[j] = sum(aPert^2)/(sum(aPert^2)+sum(a^2))  
       
     # compute p-value  
     pvalue[j] = test_for_unconfoundedness(t(X),t(Y))
  }
  mar.default= c(5, 4, 4, 2) + 0.1
  par(mar=mar.default + 3)
  plot(beta,pvalue,pch='*',col='blue',xlab=expression(beta),ylab='p-value',cex.lab=2,cex.main=2,main =paste('d=',d,',  n=',as.integer(n),collapse='',sep=''))
}

