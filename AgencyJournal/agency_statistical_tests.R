# Script for performing analysis for the Agency Journal article (adapting AIED17)
# Data output from Python Script
library(car)

# Reading in data
data = read.csv("C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/AgencyJournal/ActivitySummaryR.csv")
high_rows = data["Condition"] == 1
low_rows = data["Condition"] == 2
no_rows = data["Condition"] == 3

# Section 4.2: Impact of Agency on Learning
learning_lm = lm(RevisedNLG ~ factor(Condition), data=data)
summary(learning_lm) # The F-test conducted here is the same as the Anova Typ-II errors
t.test(data[high_rows,"RevisedNLG"], data[low_rows,"RevisedNLG"])
t.test(data[high_rows,"RevisedNLG"], data[no_rows,"RevisedNLG"])
t.test(data[low_rows,"RevisedNLG"], data[no_rows,"RevisedNLG"])

t.test(data[high_rows,"FinalGameScore"], data[low_rows,"FinalGameScore"])

# Section 4.3: Impact of Agency on Problem-Solving Behaviors
# Testing differences in rates/min of some actions in the Diagnosis phase
data = read.csv("C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/AgencyJournal/ActivitySummaryR_NoTrace.csv")
high_rows = data["Condition"] == 1
low_rows = data["Condition"] == 2
action_columns = c("PostScan.BooksAndArticles.Count",
                   "PostScan.Conversation.Count",
                   "PostScan.Scanner.Count", 
                   "PostScan.Worksheet.Count",
                   "PostScan.WorksheetSubmit.Count")
action_data = as.matrix(data[high_rows | low_rows, action_columns])
condition = data[high_rows | low_rows, "Condition"]
action_lm = lm(action_data ~ condition)
fit = manova(action_data ~ condition)
summary(fit, test="Pillai") # MANOVA test for differences in action rates from condition
