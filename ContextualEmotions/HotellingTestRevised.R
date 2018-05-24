# Revised script for analysis of contextual emotions during Crystal Island (Learning and Instruction article)
# This script uses multivariate over emotions for each action difference
library(Hotelling)

data = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/Publications/LI-Contextual-Emotions/Data/DesiredWindows_Reduced.csv")
dim(data)

emotions_single_context_hotelling <- function(keyword, prefix, pos_key, neg_key, data, comp_type="CountRate"){
  emotions = c("Frustration", "Confusion", "Joy")
  specific_data = data[data[,paste(keyword,"Active", sep="")] == 1,]
  
  Yp = NULL
  Yn = NULL
  
  for(emotion in emotions){
    pos_col = paste0(prefix, keyword, pos_key, ".", emotion, "Evidence.", comp_type)
    positive_rows = specific_data[, pos_col]
    Yp = cbind(Yp, positive_rows)
    
    neg_col = paste0(prefix, keyword, neg_key, ".", emotion, "Evidence.", comp_type)
    negative_rows = specific_data[, neg_col]
    Yn = cbind(Yn, negative_rows)
  }
  
  h = hotelling.test(Yp - Yn, matrix(0, dim(Yp)[1], dim(Yp)[2]))
  return(list(context=keyword, 
              statistic=round(h$stats$statistic,3), 
              df1=h$stats$df[1], 
              df2=h$stats$df[2], 
              p=round(h$pval,3)))
}

# Generating main results table
comparison_type="DurationProp"
emotions_single_context_hotelling(keyword="Scan", prefix="After.", pos_key="Positive", neg_key="Negative", data=data, comp_type = comparison_type)
emotions_single_context_hotelling(keyword="Book", prefix="During.", pos_key="Relevant", neg_key="Irrelevant", data=data, comp_type = comparison_type)
emotions_single_context_hotelling(keyword="Submission", prefix="After.", pos_key="Correct", neg_key="Incorrect", data=data, comp_type = comparison_type)

# Post hoc tests for scans
scan_data = data[data[,"ScanActive"]==1, ]
Yp = NULL
Yn = NULL
for(e in c("Frustration", "Confusion", "Joy")){
  pos_rows = scan_data[,paste0("After.ScanPositive.", e, "Evidence.", comparison_type)]
  Yp = cbind(Yp, pos_rows)
  neg_rows = scan_data[,paste0("After.ScanNegative.", e, "Evidence.", comparison_type)]
  Yn = cbind(Yn, neg_rows)
  print(e)
  print(t.test(pos_rows - neg_rows))
  print(mean(pos_rows - neg_rows))
  print(sd(pos_rows - neg_rows))
}

# Manual calculation of Hotelling statistic
# Yd = Yp - Yn
# samp_mean = apply(Yd, 2, mean)
# samp_cov = cov(Yd)
# n = dim(Yd)[1]
# T2.stat = (n * t(samp_mean) %*% solve(samp_cov) %*% samp_mean)[1,1]

# Generating counts for the n of the results table
sum(data[,"ScanActive"])
sum(data[,"BookActive"])
sum(data[,"SubmissionActive"])
