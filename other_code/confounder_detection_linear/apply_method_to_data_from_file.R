apply_method_to_data_from_file=function(path_data){
  # input:                      Full path of the directory in which the data file is located. File must contain a data matrix n x (d+1)
  #                             of n samples from X_1,...,X_d,Y, where Y is the target variable and X_1,..,X_d the potential causes.
  #                             The command line prompt asks how the numbers are separated.
  #                             Later, the command line prompt asks whether some of the variables X_j should be dropped.
  #                             The purpose is two-fold: First, it allows to drop irrelevant variables.
  #                             Second, it allows to test the algorithm on data where the confounder Z is known: whenever one decides to drop
  #                             exactly one of the X_j, the algortithm computes the confounding strength under the assumption that this X_j is the 
  #                             confounder Z. In the data sets in the folder 'optical_device', for instance, the confounder is the 10th column
  #                             (except for the file ending with 'random_image_section').
  #
  #                             
  #
  # command line output:        (1) estimated confounding parameters beta_est and eta_est 
  #                             (2) true confounding strength beta (for the case where one variable X_j has been dropped, 
  #                                  under the simplified assumption that X_j is the confounder Z)
  #
  # function output:            n x (d+1) data matrix where each row consists of values attained by X_1,...,X_d,Y
  #
  #  
  
  
  path = '/home/dk/Dropbox/projects/causality/multivariate/code/other_code/confounder_detection_linear'
  setwd(path)
  source('estimate_confounding_via_kernel_smoothing.R')

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
  
  # read file and store values of X and Y
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
    cxx=cov(X,use='pairwise.complete.obs')
    standarddev=sqrt(diag(diag(cxx)))
    norma=solve(standarddev)
    X= X %*% norma 
  }
  
  # compute regression vector for the full set of predictor variables X
  a_total=(solve(cov(X,use='pairwise.complete.obs')) %*% cov(X,Y,use='pairwise.complete.obs'))
  print('a_total= ')
  print(a_total)
  
  # choose components to drop
  xdropped=as.integer(strsplit(readline(prompt='which components should be dropped (space separated list)? '),split=' ')[[1]])
  xselect=setdiff(1:d,xdropped)
  Xred=X[,xselect]
  Y=D[,d+1]
  d_red=length(xselect)
  cxxred=cov(Xred,use='pairwise.complete.obs')
  cxxredinvers=solve(cxxred)
  cxyred=cov(Xred,Y,use='pairwise.complete.obs')
  a_hat=cxxredinvers %*% cxyred
  
  # compute confounding strength under the assumption that the dropped component (if it is only one variable) is the confounder 
  if (length(xdropped)==1){
    a_red=a_total[xselect]
    print(xdropped)
    Z=X[,xdropped]
    sigmaZ=sd(Z,na.rm=T)
    Z=Z/sigmaZ
    c=a_total[xdropped]*sigmaZ  
    b=cov(Xred,Z,use='pairwise.complete.obs')
    error_vector=cxxredinvers %*% b *c
    s=sum((error_vector)^2)  
    sqnorm_a=sum(a_red^2)
    beta_true=s/(sqnorm_a+s)
    
    # estimate explanatory power
    eta_true=sum(b^2)
    
    # output true parameters 
    cat('beta_true=',beta_true,'\n')
    cat('eta_true=',eta_true,'\n')
    
    # compute spectral measure with respect to Sigma_EE  
    cee=cxxred-b %*% t(b)
    spectral_dec_E=eigen(cee)
    spectrum_E=spectral_dec_E$values
    eigenvectors_E=spectral_dec_E$vectors  
  }
  
  # display correlations
  print('correlations between X (reduced set) and Y: ')
  print(cor(Xred,Y,use='pairwise.complete.obs'))
 
  for (j in 1:d_red){
    print(cor.test(Xred[,j],Y,use='pairwise.complete.obs'))
  } 
  
  # compute confounding caused by dropping one component
  if (length(xdropped)==1){
     cat('true confounding parameters:','\n')
     cat('beta_true = ',beta_true,'\n')
     cat('eta_true = ',eta_true,'\n')     
  }
  cat('\n')
  cat('estimated confounding parameters= ','\n')
  lambda_est = estimate_confounding_via_kernel_smoothing(Xred,Y)
  beta_est = lambda_est[1]
  eta_est = lambda_est[2]
  cat('beta_est = ',beta_est,'\n')
  cat('eta_est = ',eta_est,'\n')
  cat('\n')
  
  cat('The user shoud be warned that inferring causal relations from purely observational data is a difficult enterprise. I disrecommend to use
      this software without understanding its highly idealized model assumptions.')
  
  # output data for further use
  data = cbind(X,Y)
}
