plot_true_vs_estimated_confounding = function(d,l,n,runs){
  
 #  This is the function producing the plots in Figure 4 of the arXiv paper 
 #  'Detecting non-causal artifacts in multivariate
 #  linear regression models' by Dominik Janzing and Bernhard Schoelkopf, 2018.   
 #  input:  d (dimension)
 #          l (number of sources) 
 #          n (sample size)
 #  output: 1) plot of p-value of the non-confounding test for different values of beta for 'runs' many instances 
 #          2) result of correlation test between true beta and estimated beta   
 #  copyright: Dominik Janzing, 2018.   

 source('~/nonlinngam/code/confounder_detection_independent_sources/estimate_theta.R') 
 
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
     theta[j] = sigma_c^2/sigma_a^2
     c = rnorm(l,0,sigma_c)
     # generate Y
     Y = a %*% X + c %*% S
     
     # compute confounding strength. First compute the perturbation of the regression vector a caused by confounding
     aPert = t(ginv(M)) %*% c
     beta[j] = sum(aPert^2)/(sum(aPert^2)+sum(a^2))  
       
     # estimate theta and then beta
     Cxx = cov(t(X))
     Conxx = solve(Cxx)
     tauConxx = sum(diag(Conxx))/d
     Cxy = cov(t(X),t(Y))
     aHatEst = solve(Cxx) %*% Cxy     
     theta_est[j] = estimate_theta(Cxx,aHatEst)
     beta_est[j] =  1/(1+1/(tauConxx*theta_est[j]))
  }
  # plot results 
  plot(beta,beta_est,pch='*',col='blue')
  # output correlations between true and estmate beta 
  cor.test(beta,beta_est)
}

