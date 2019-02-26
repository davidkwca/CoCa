my_path = '~/bayes/other_code/confounder_detection_independent_sources/'
source(paste(my_path, 'test_for_unconfoundedness.R', sep=''))
source(paste(my_path, 'estimate_theta.R', sep=''))

library(psych)

args = commandArgs(trailingOnly=TRUE)
file_name = args[1]
D=read.csv(file_name,header=F,sep=',',skip=0,stringsAsFactors=FALSE)

D=apply(D,c(1,2),as.numeric)
d=dim(D)[2]-1

X=D[,1:d]
Y=D[,d+1]


a_total=(solve(cov(X,use='pairwise.complete.obs')) %*% cov(X,Y,use='pairwise.complete.obs'))
a_total=(solve(cov(X,use='pairwise.complete.obs')) %*% cov(X,Y,use='pairwise.complete.obs'))


cxx=cov(X,use='pairwise.complete.obs')
cxxinvers=solve(cxx)
cxy=cov(X,Y,use='pairwise.complete.obs')

a_hat=cxxinvers %*% cxy

tauConxx = sum(diag(cxxinvers))/(d-1)
aHatEst = cxxinvers %*% cxy     
theta_est = estimate_theta(cxx,aHatEst) 
beta_est =  1/(1+1/(tauConxx*theta_est))
pvalue = test_for_unconfoundedness(X,Y)

cat(beta_est, ',', pvalue)
