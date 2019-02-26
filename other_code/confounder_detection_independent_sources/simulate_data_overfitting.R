simulate_data_overfitting = function(d,l,n,runs){
 #setwd('D:')  
 library(MASS)  
  
 source('~/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/generate_covariances.R')
 source('~/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/estimate_par.R')
 source('~/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/density.R')
 source('~/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/estimate_theta.R')
  
  #source('/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/generate_covariances.R')
  #source('/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/estimate_par.R')
  #source('/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/density.R')
  #source('/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/estimate_theta.R')
  #source('/SVN_Dominik/code/R_code/many_variable_confounding/independent_sources_assumption/test_for_unconfoundedness.R')
 
  pvalue = rep(0,runs)
  for (j in 1:runs){
     cat('j = ',j,'\n')  
     # generate sources
     S = matrix(rnorm(n*l,0,1),nrow=l,ncol=n)
     # generate mixing matrix
     M = matrix(rnorm(d*l,0,1),nrow=d,ncol=l)
     #M = diag(1:d)
     # generate X
     X = M %*% S
     sigma_a =  1
     # generate structure coefficients
     a = rnorm(d,0,sigma_a)
     # generate Y
     Y = a %*% X + rnorm(n,0,1)
       
     Cxx = cov(t(X))
     Conxx = solve(Cxx)
     #tauConxx = sum(diag(Conxx))/d
     Cxy = cov(t(X),t(Y))
     aHatEst = solve(Cxx) %*% Cxy     
     #theta_est[j] = estimate_theta(Cxx,aHatEst) 
     #beta_est[j] =  1/(1+1/(tauConxx*theta_est[j]))
     pvalue[j] = test_for_unconfoundedness(t(X),t(Y))
  }
 hist(pvalue,breaks=20,freq=T,col='blue',main=paste('sample size = ',n))
}

