my_path = '~/bayes/other_code/confounder_detection_linear/'
source(paste(my_path, 'estimate_confounding_via_kernel_smoothing.R', sep=''))

args = commandArgs(trailingOnly=TRUE)
file_name = args[1]
D=read.csv(file_name,header=F,sep=',',skip=0,stringsAsFactors=FALSE)

D=apply(D,c(1,2),as.numeric)
d=dim(D)[2]-1

X=D[,1:d]
Y=D[,d+1]

## xselect=setdiff(1:d,10)
## X=X[,xselect]


## cxx=cov(X,use='pairwise.complete.obs')
## standarddev=sqrt(diag(diag(cxx)))
## norma=solve(standarddev)
## X= X %*% norma 


lambda_est = estimate_confounding_via_kernel_smoothing(X,Y)
beta_est = lambda_est[1]

cat(beta_est)
