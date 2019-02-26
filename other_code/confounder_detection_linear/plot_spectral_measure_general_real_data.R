plot_spectral_measure_general_real_data=function(path_data){
  # input: full absolute path of the data directory
  # output: (1) (display) bar plot of the weights of the spectral measure of a_hat with respect to Sigma_XX
  #         (2) (function output) data matrix X Y for further use
  
  #  choose data
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
 
  # display structural coefficients
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
    
  # compute confounding strength
  if (length(xdropped)==1){
    a_red=a_total[xselect]
    Z=X[,xdropped]
    sigmaZ=sd(Z)
    Z=Z/sigmaZ
    c=a_total[xdropped]*sigmaZ  
    b=cov(Xred,Z,use='pairwise.complete.obs')
    error_vector=cxxredinvers %*% b *c
    s=sum((error_vector)^2)  
    sqnorm_a=sum(a_red^2)
    beta_true=s/(sqnorm_a+s)
    
    # estimate explanatory power
    eta=sum(b^2)
  }
  
  # compute spectral measure
  spectral_dec=eigen(cxxred)
  eigenvectors=spectral_dec$vectors
  weights=(t(a_hat) %*% eigenvectors)^2
  weights=weights/sum(weights)
  spectrum=spectral_dec$values
  
  # print spectral measure 
  gaps=rev(spectrum[1:d_red]-c(spectrum[2:d_red],0))
  width_bar=spectrum[1]/(5*d_red)
  space_bars=gaps*5*d_red/spectrum[1] -1
  truncated_file_name=strsplit(file_list[file_number],split='[.]')[[1]][1]
  title=paste(strtrim(truncated_file_name,15),'; dropped:',xdropped)
  if (length(xdropped)==1)
      title=paste(title,'; beta=',round(beta_true,digits=2),' ; eta=',round(eta,digits=2))
  pl=barplot(rev(weights),col='blue',width=width_bar,space=space_bars,xlim=c(0,spectrum[1]),main=title)
  axis(1,at=pl,labels=round(rev(spectrum),digits=2))
  
  # output data for further use
  data = cbind(X,Y)
}