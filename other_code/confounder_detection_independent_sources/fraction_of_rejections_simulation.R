fraction_of_rejections_simulation = function(d,l,n,runs,alpha){
 
  
  #  This is the function producing the plot in Figure 5 of the arXiv paper 
  #  'Detecting non-causal artifacts in multivariate
  #  linear regression models' by Dominik Janzing and Bernhard Schoelkopf, 2018.   
  #  inputs: d (dimension)
  #          l (number of sources) 
  #          n (sample size)
  #          alpha (significance level)
  #  output: plot true confounding beta vs fraction of rejections for 'runs' many instances 
  #          
  #  copyright: Dominik Janzing, 2018.   
  
  # uses 'test_for_unconfoundedness.R'  
  
  source('~/nonlinngam/code/confounder_detection_independent_sources/test_for_unconfoundedness.R')
  
  library(MASS)  
 
  beta = rep(0,runs)
  pvalue = rep(0,runs)
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
 
  # compute percent of rejected samples for each interval of beta
  beta_discrete = ceiling(beta * 10)
  fraction = rep(0,10)
  for (i in 1:10){
    indices = which(beta_discrete == i)
    fraction[i] = length(which(pvalue[indices]<alpha))/length(indices)
  }
  # set plot parameters and plot
  mar.default= c(5, 4, 4, 2) + 0.1
  par(mar=mar.default + 3)
  midpoints = barplot(fraction,names.arg=NULL,axisnames = T,xlab = expression(beta),ylab='fraction of rejections',col='blue',cex.lab=1.5, ylim=c(0,1),main=paste(expression(alpha),'=',alpha))
  ticks = (midpoints[2:10] + midpoints[1:9])/2
  lastTick = midpoints[10] +  (midpoints[10]-midpoints[9])/2 
  ticks = c(0,ticks,lastTick)
  axis(side = 1, at=ticks, labels=0:10/10)
}

