try_general_real_data=function(path_data){

  
  #  This is the function for detecting and estimating confounding as described in the arXiv paper 
  #  'Detecting non-causal artifacts in multivariate
  #  linear regression models' by Dominik Janzing and Bernhard Schoelkopf, 2018.   
  
  #  input: path of the directory in which the data files are stored. Format:
  #         text or csv file, potentially with header, with (n x d+1) matrix whose first d columns are n instances
  #         of X_1,...,X_d and the d-th columns is Y  
  #         
  #  output: 1) beta_true (if one has chosen to drop one X_j when asked later 
  #                      in the command line prompt), true confounding is defined by taking  
  #                      this X_j as confounder
  #          2) beta_est (estimated confounding strength)
  #          3) p-value of the test for non-confounding    
  #          4) (n x d+1) data matrix (X,Y) for further use
  #          
  #  copyright: Dominik Janzing, 2018.   
  
  # uses 'test_for_unconfoundedness.R' and 'estimate_theta.R'
   
  source('test_for_unconfoundedness.R')
  source('estimate_theta.R')
  
  library(psych)
 
  # display the files in the respective data directory
  directory_list=list.dirs(path_data,recursive=F,full.names=F)
  for (j in 1:length(directory_list))
    cat(j, directory_list[[j]],'\n')
  dir_number=as.integer(readline(prompt='choose a number '))
  full_path=paste(path_data,'/',directory_list[[dir_number]],collapse='',sep='')
  file_list=list.files(full_path,full.names=F)
  for (j in 1:length(file_list))
    cat(j, file_list[[j]],'\n')
  
  # ask user for the file number to be used
  file_number=as.integer(readline(prompt='choose a number '))
  file_name=paste(full_path,'/',file_list[file_number],collapse='',sep='')
  # ask user about format of the file
  header = readline(prompt='header? y/n ')
  separator = readline(prompt='separator= ')
  
  # read data
  if (header=='y')
   D=read.csv(file_name,header=T,sep=separator,skip=0,stringsAsFactors=FALSE)
  else 
   D=read.csv(file_name,header=F,sep=separator,skip=0,stringsAsFactors=FALSE)
  
  D=apply(D,c(1,2),as.numeric)
  d=dim(D)[2]-1
  
  X=D[,1:d]
  Y=D[,d+1]
  
  # normalization of X
  answer=readline('should X be normalized? y/n ')
  if (answer=='y'){
    print(X)
    cxx=cov(X,use='pairwise.complete.obs')
    standarddev=sqrt(diag(diag(cxx)))
    norma=solve(standarddev)
    X= X %*% norma 
  }
  
  # compute regression vector
  a_total=(solve(cov(X,use='pairwise.complete.obs')) %*% cov(X,Y,use='pairwise.complete.obs'))
  print('a_total= ')
  print(a_total)

  # display structural coefficients
  cat('the vector of structure coefficients reads\n')
  a_total=(solve(cov(X,use='pairwise.complete.obs')) %*% cov(X,Y,use='pairwise.complete.obs'))
  cat('a_total= ',a_total,'\n\n')
  
  # choose components to drop
  cat('if you now choose exactly one X_j as hidden variable confounding is computed with this variable as confounder\n')
  xdropped=as.integer(strsplit(readline(prompt='which components should be dropped (space separated list)? '),split=' ')[[1]])
  xselect=setdiff(1:d,xdropped)
  Xred=X[,xselect]
  Y=D[,d+1]
  d_red=length(xselect)
  cxxred=cov(Xred,use='pairwise.complete.obs')
  cat('\n')  

  cxxredinvers=solve(cxxred)
  cxyred=cov(Xred,Y,use='pairwise.complete.obs')

  a_hat=cxxredinvers %*% cxyred
  
  # compute confounding strength
  if (length(xdropped)==1){
    a_red=a_total[xselect]
    Z=X[,xdropped]
    sigmaZ=sd(Z,na.rm=T)
    Z=Z/sigmaZ
    c=a_total[xdropped]*sigmaZ  
    b=cov(Xred,Z,use='pairwise.complete.obs')
    error_vector=cxxredinvers %*% b *c
    s=sum((error_vector)^2)  
    sqnorm_a=sum(a_red^2)
    beta_true=s/(sqnorm_a+s)
    
   
    # output true beta
    cat('beta_true = ',beta_true,'\n\n')  
  }
  tauConxx = sum(diag(cxxredinvers))/(d-1)
  aHatEst = cxxredinvers %*% cxyred     
  theta_est = estimate_theta(cxxred,aHatEst) 
  beta_est =  1/(1+1/(tauConxx*theta_est))
  pvalue = test_for_unconfoundedness(X,Y)
    
  cat('beta_est = ',beta_est,'\n')
  cat('pvalue = ',pvalue,'\n\n')
  
  cat('The user shoud be warned that inferring causal relations from purely observational data is a difficult enterprise. I disrecommend to use
      this software without understanding its highly idealized model assumptions.')
  
}
