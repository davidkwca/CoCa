simulation=function(d,sample_size,runs){
  
  # input: d (dimension), sample_size (sample size), runs (number of runs)
  # output: scatter plot that shows the relation between true and estimated confounding strength beta. The number of points is given by the 
  # parameter 'runs', 

  path = '~/nonlinngam/code/confounder_detection_linear'
  setwd(path)
  source('estimate_confounding_via_kernel_smoothing.R')
  
  beta = rep(runs,0)
  eta = rep(runs,0)
  beta_est = rep(runs,0)
  eta_est = rep(runs,0)
  for (i in 1:runs){ 
    # randomly draw model parameters
    
    # length of vector a
    r_a = runif(1,0,1)
    # length of vector b
    r_b = runif(1,0,1)
    # strength of influence of confounder on target variable Y
    c = runif(1,0,1)
    # draw random vectors a and b 
    a = rnorm(d,0,1)
    b = rnorm(d,0,1)
    a = a/sqrt(sum(a^2)) * r_a
    b = b/sqrt(sum(b^2)) * r_b
  
    # generate samples of the noise vector E 
    E = matrix(rnorm(sample_size*d,0,1), sample_size, d)
    random_orthogonal = randortho(d)
    random_matrix = matrix( rnorm(d*d,0,1), d,d)
    E = E %*% random_matrix
    
    # generate samples of the confounder Z and the noise term NY for the target variable Y
    Z = rnorm(sample_size, 0, 1)
    NY = rnorm(sample_size, 0, 1)  
  
    # compute X and Y via linear structural equations
    X = E +  Z %*% t(b)
    Y = c * Z + X %*% a + NY
  
    # compute confounding parameters
    SigmaEE = t(random_matrix) %*% random_matrix
    SigmaXX = SigmaEE + b %*% t(b)
    confounding_vector = c * solve(SigmaXX) %*% b
    sq_length_cv=sum(confounding_vector^2)
    beta[i] =  sq_length_cv / (r_a^2 + sq_length_cv)
    eta[i] = r_b^2
   
    # estimate both confounding parameters
    parameters = estimate_confounding_via_kernel_smoothing(X,Y)
    beta_est[i]=parameters[1]
    eta_est[i]=parameters[2]
  }

# plot results
mar.default= c(5, 4, 4, 2) + 0.1
par(mar=mar.default + 3)
plot(beta,beta_est,pch='*',col='blue',xlab=expression(beta),ylab=expression(hat(beta)),cex.lab=2,cex.main=2,main =paste('d=',d,',  n=',as.integer(sample_size),collapse='',sep=''))

# test correlation between true and estimated confounding strength
cor.test(beta,beta_est)
}