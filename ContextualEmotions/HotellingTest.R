# Script for analysis of contextual emotions during Crystal Island (Learning and Instruction article)
library(Hotelling)

data = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/Publications/LI-Contextual-Emotions/Data/DesiredWindows_Zeroed.csv")
dim(data)

context_hotelling <- function(e, data, comp_type="DurationProp"){
  scans_pos = data[,paste("After.ScanPositive.", e, "Evidence.", comp_type, sep="")]
  books_rel = data[,paste("During.BookRelevant.", e, "Evidence.", comp_type, sep="")]
  subs_cor = data[,paste("After.SubmissionCorrect.", e, "Evidence.", comp_type, sep="")]
  Yp = cbind(scans_pos, books_rel)
  
  scans_neg = data[,paste("After.ScanNegative.", e, "Evidence.", comp_type,  sep="")]
  books_irr = data[,paste("During.BookIrrelevant.", e, "Evidence.", comp_type, sep="")]
  subs_inc = data[,paste("After.SubmissionIncorrect.", e, "Evidence.", comp_type, sep="")]
  Yn = cbind(scans_neg, books_irr)
  
  h = hotelling.test(Yp - Yn)
  h$stats$statistic
  h$stats$df
  h$pval
  
  scan_diff = scans_pos - scans_neg
  book_diff = books_rel - books_irr
  subs_diff = subs_cor - subs_inc
  return(list(emotion=e, 
              statistic=round(h$stats$statistic,3), 
              df1=h$stats$df[1], 
              df2=h$stats$df[2], 
              p=round(h$pval,3),
              scan_mdif=round(mean(scan_diff),3),
              scan_std=round(sd(scan_diff), 2),
              book_mdif=round(mean(books_rel - books_irr),3),
              book_std=round(sd(book_diff), 2),
              subs_mdif=round(mean(subs_cor - subs_inc),3),
              subs_std=round(sd(subs_diff), 2)))
}

# Generating Results table of testing each emotion's difference of action outcome
# Table 3.1
def_comp_type = "DurationProp"
cdf = NULL
for(emotion in c("Anger", "Frustration", "Disgust", "Surprise", "Contempt", "Confusion", "Joy", "Fear", "Sadness")){
  new_data <- context_hotelling(emotion, data, comp_type=def_comp_type)
  cdf <- rbind(cdf, new_data)
  rownames(cdf)[nrow(cdf)] <- emotion
}
cdf

# Post-hoc testing for Surprise, since significant difference on the Hotelling T2 level
# Table 3.2
e = "Surprise"
scans_pos = data[,paste("After.ScanPositive.", e, "Evidence.", def_comp_type, sep="")]
books_rel = data[,paste("During.BookRelevant.", e, "Evidence.", def_comp_type, sep="")]
subs_cor = data[,paste("After.SubmissionCorrect.", e, "Evidence.", def_comp_type, sep="")]

scans_neg = data[,paste("After.ScanNegative.", e, "Evidence.", def_comp_type,  sep="")]
books_irr = data[,paste("During.BookIrrelevant.", e, "Evidence.", def_comp_type, sep="")]
subs_inc = data[,paste("After.SubmissionIncorrect.", e, "Evidence.", def_comp_type, sep="")]

t.test(scans_pos - scans_neg)
t.test(books_rel - books_irr)
t.test(subs_cor - subs_inc)

# Post-hoc testing for Fear, since significant difference on the Hotelling T2 level
# Table 3.3
e = "Fear"
scans_pos = data[,paste("After.ScanPositive.", e, "Evidence.", def_comp_type, sep="")]
books_rel = data[,paste("During.BookRelevant.", e, "Evidence.", def_comp_type, sep="")]
subs_cor = data[,paste("After.SubmissionCorrect.", e, "Evidence.", def_comp_type, sep="")]

scans_neg = data[,paste("After.ScanNegative.", e, "Evidence.", def_comp_type,  sep="")]
books_irr = data[,paste("During.BookIrrelevant.", e, "Evidence.", def_comp_type, sep="")]
subs_inc = data[,paste("After.SubmissionIncorrect.", e, "Evidence.", def_comp_type, sep="")]

t.test(scans_pos - scans_neg)
t.test(books_rel - books_irr)
t.test(subs_cor - subs_inc)


