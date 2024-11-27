


# mean vector of a certain distribution
#takes one distribution vector of certain step segmentation and fit the vector to 2-components GMM, output a mean vector contains either 1 or 2 means, univarient

GMMMeanVec <- function(DataVec){
  
  # Fit the distribution to GMM
  gmm_model <- Mclust(DataVec, G = 1:2)
  
  # Print a summary of the model
  # summary(gmm_model,parameter = TRUE)
  # plot(gmm_model, what = "BIC" )
  # plot(gmm_model, what = "density" )
  
  mean_vec <- as.vector(gmm_model$parameters$mean) # vec contains 1 or 2 objects
  
  # if the subtraction of the two components means < 0.1, merge them to one, and output a new mean_vec
  if(length(mean_vec) == 2){
    
    if (mean_vec[2] - mean_vec[1] < 0.1)
      gmm_model <- Mclust(DataVec, G = 1)
    mean_vec <- as.vector(gmm_model$parameters$mean)
  } 
  
  
  print(mean_vec)
  
}





# mean matrix of distributions by steps
# feed a matrix contains repetition r, step t, bestMeanProb (and averageReward), output mean matrix by step
GMMMeanMatrix <- function(DataMatrix){
  
  
  t_list <- seq(0,horizon-step,step) # step and horizon is pre-defined
  components_mean <- matrix(NA, length(t_list), 3) # col1 summed step, col2 mean1, col3 mean2
  components_mean[ ,1] <- t_list + step # empty matrix for step, component 1 and  component 2
  
  
  for (x in t_list) {
    
    sum_facet <- DataMatrix %>% filter(t > x &  t < (x+step+1)) 
    
    dist_facet <- sum_facet %>% 
      # the new dataset is copied from the original dataset
      #dplyr::group_by(sigma, theta, r) %>% 
      dplyr::group_by(r) %>% 
      summarise( # summarising the data set with the group structure considered.
        # the following two become new variables
        bestMeanProb_globalMean = mean(bestMeanProb), 
        averageReward_globalMean = mean(averageReward),
        # HotStoveSusceptibility = mean(HotStoveSusceptibility)
      )
    
    
    bestMeanVec <- dist_facet$bestMeanProb_globalMean
    
    meanOfThisT <- GMMMeanVec(bestMeanVec)
    
    
    if (length(meanOfThisT) == 1){
      
      components_mean[match(x, t_list) , 3] <- meanOfThisT
      
    }
    
    else if(length(meanOfThisT) == 2){
      
      components_mean[match(x, t_list), 2:3] <- meanOfThisT
      
    }
    
    # if meanOfThisT has 1 object, copy the object to the NAs. If the distribution has 1 component, then the mean is copied to all components
    NA_index <- which(is.na(components_mean[ , 2]))
    
    test_vec<- as.vector(components_mean[ , 3])
    
    components_mean[ , 2][NA_index]  <- test_vec[NA_index] #col1 summed step, col2 mean1, col3 mean2
    
  }
  return(components_mean)
}



