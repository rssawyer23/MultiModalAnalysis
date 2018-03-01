# Script for analyzing the output of trajectory distance preprocessing (EDM 2018)
# LOADING AND BASIC VISUALIZATION OF DATA
full_output_path = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/GoldenPathDistancesT.csv"
data = read.csv(full_output_path)
names(data)
hist(data[,"Duration"])
hist(data[,"Average"])
plot(data[,"Duration"], data[,"Average"])
plot(data[,"Average"], data[,"NLG"])

# SUMMARY OF DISTANCE FOR FULL AGENCY
hist(data[data[,"Condition"]==1,"Average"])
mean(data[data[,"Condition"]==1,"Average"])
sd(data[data[,"Condition"]==1, "Average"])

# SUMMARY OF DISTANCE FOR PARTIAL AGENCY
hist(data[data[,"Condition"]==0,"Average"])
mean(data[data[,"Condition"]==0,"Average"])
sd(data[data[,"Condition"]==0, "Average"])

hist(data[,"Max"])

# SUMMARY STATISTICS BY CONDITION
summary(data[,"Duration"]/60)
mean(data[, "Duration"]/60)
sd(data[,"Duration"]/60)


# KEY CORRELATION TESTS
cor.test(data[,"NLG"], data[,"Average"])  # Distance to gold path and NLG significantly correlated
cor.test(data[data[,"Condition"]==1,"Average"], data[data[,"Condition"]==1,"NLG"])  # Full subgroup shows similar correlation
cor.test(data[data[,"Condition"]==0,"Average"], data[data[,"Condition"]==0,"NLG"])  # Partial subgroup shows somewhat similar correlation
cor.test(data[,"NLG"], data[,"Duration"])  # Duration not significantly correlated with NLG
cor.test(data[,"Average"], data[,"Duration"])  # Duration and distance not correlated
scaled_avg = scale(data[,"Average"])
lm1 = lm(NLG~Average, data=data)
summary(lm1)

# CORRELATION TESTS WITH MAX
cor.test(data[,"NLG"], data[,"Max"])
cor.test(data[data[,"Condition"]==1,"Max"], data[data[,"Condition"]==1,"NLG"])
cor.test(data[data[,"Condition"]==0,"Max"], data[data[,"Condition"]==0,"NLG"])
cor.test(data[,"Max"], data[,"Duration"])


# TESTING TO SEE IF FULL AND PARTIAL AGENCY CAN BE IDENTIFIED AS DIFFERENT DISTRIBUTIONS
t.test(data[data[,"Condition"]==1,"Average"],data[data[,"Condition"]==0,"Average"])
full_distances = data[data[,"Condition"]==1,"Average"]
part_distances = data[data[,"Condition"]==0,"Average"]
wilcox.test(full_distances, part_distances)

# LOADING BASELINE FILE FOR COMPARISONS
baseline_path = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesPC1.csv"
b.data = read.csv(baseline_path)
names(b.data)
cor.test(b.data[1:101,"NLG"], b.data[1:101,"FinalGameScore"])  # Significant correlation between NLG and Game Score, but not as strong as Distance
b.lm = lm(NLG~C.A.BooksAndArticles + 
            C.A.Conversation +
            C.A.PlotPoint +
            C.A.Scanner +
            C.A.Worksheet +
            C.A.WorksheetSubmit +
            Duration,
          data=b.data)
summary(b.lm)

gp_PC = b.data[dim(b.data)[1],"PC1"]  # Getting final row PC filter value for the golden path
gp_dur = b.data[dim(b.data)[1],"Duration"]
b.data["FinalDist"] = sqrt((b.data[,"PC1"] - gp_PC)^2)  # Calculating the distance with the final values against golden path
b.data["DurationDist"] = b.data[,"Duration"] - gp_dur # Calculating the difference in duration between student and golden path

b.data.ogp = b.data[1:101,]
b.data.ogp[,"TestSubject"] = factor(b.data.ogp[,"TestSubject"])
cor.test(b.data.ogp[,"FinalDist"], data[,"Average"])  # Showing that final point distance and sequential distance are correlated
cor.test(b.data.ogp[,"FinalDist"], b.data.ogp[,"NLG"])  # But that this final distance is not correlated with NLG

cor.test(b.data.ogp[,"FinalDist"], b.data.ogp[,"FinalGameScore"])
cor.test(b.data.ogp[,"PC1"], b.data.ogp[,"NLG"])
cor.test(b.data.ogp[,"PC1"], b.data.ogp[,""])

#Checking subjects are aligned before combining dataframes
sum(data[,"TestSubject"] == b.data.ogp[,"TestSubject"]) == dim(data)[1]
sum(data[,"TestSubject"] == b.data.ogp[,"TestSubject"]) == dim(b.data.ogp)[1]

#f.data = cbind(data, b.data.ogp)
f.data = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevised.csv")
plot(f.data[,"RevisedDuration"], f.data[,"Average"], xlab="Total Gameplay Duration", ylab="Temporal Gold Path Distance")
f.data["DurationDist"] = f.data["RevisedDuration"] - gp_dur
plot(abs(f.data[,"DurationDist"]), f.data[,"Average"])

summary(f.data[,"RevisedDuration"]/60.0)
sd(f.data[,"RevisedDuration"]/60.0)

cor.test(f.data[,"DurationDist"],f.data[,"Average"]) # Showing distance metric not correlated to duration distance between golden path
cor.test(f.data[f.data["DurationDist"]>0,"DurationDist"], f.data[f.data["DurationDist"]>0,"Average"])
cor.test(f.data[f.data["DurationDist"]<0,"DurationDist"], f.data[f.data["DurationDist"]<0,"Average"])

#TESTING DURATION BY CONDITION
cor.test(f.data[,"RevisedDuration"], f.data[,"Average"])
cor.test(f.data[f.data[,"Condition"]==1,"RevisedDuration"], f.data[f.data[,"Condition"]==1, "Average"])
cor.test(f.data[f.data[,"Condition"]==0,"RevisedDuration"], f.data[f.data[,"Condition"]==0, "Average"])
summary(lm(Average~scale(Duration) + I(scale(Duration)^2), data=f.data[f.data[,"Condition"]==1,]))
summary(lm(Average~scale(abs(Duration)) + I(scale(abs(Duration))^2), data=f.data[f.data[,"Condition"]==1,]))

summary(lm(Average~Duration + abs(DurationDist), data=f.data))

cor.test(f.data[,"Average"], f.data[,"FinalGameScore"])

cor.test(f.data[,"FinalDist"], f.data[,"NLG"])  # But that this final distance is not correlated with NLG
cor.test(f.data[f.data[,"Condition"]==1,"FinalDist"], f.data[f.data[,"Condition"]==1,"NLG"])
cor.test(f.data[f.data[,"Condition"]==0,"FinalDist"], f.data[f.data[,"Condition"]==0,"NLG"])

summary(f.data[,"FinalGameScore"])
sd(f.data[,"FinalGameScore"])
hist(f.data[,"FinalGameScore"])
shapiro.test(f.data[,"FinalGameScore"])
t.test(f.data[f.data[,"Condition"]==1, "FinalGameScore"], f.data[f.data[,"Condition"]==0, "FinalGameScore"])
cor.test(f.data[,"FinalGameScore"], f.data[,"NLG"])
cor.test(f.data[f.data[,"Condition"]==1,"FinalGameScore"], f.data[f.data[,"Condition"]==1,"NLG"])
cor.test(f.data[f.data[,"Condition"]==0,"FinalGameScore"], f.data[f.data[,"Condition"]==0,"NLG"])

# TESTING IF THE FINAL DISTANCE CAN DISTINGUISH BETWEEN CONDITIONS AS SEQUENTIAL DISTANCE DOES
wilcox.test(f.data[f.data[,"Condition"]==1,"FinalDist"], f.data[f.data[,"Condition"]==0,"FinalDist"]) # Does not reveal signifance
wilcox.test(f.data[f.data[,"Condition"]==1,"FinalGameScore"], f.data[f.data[,"Condition"]==0,"FinalGameScore"]) # Does not reveal signifance
wilcox.test(f.data[f.data[,"Condition"]==1,"Duration"], f.data[f.data[,"Condition"]==0,"Duration"])

f.lm = lm(NLG~C.A.BooksAndArticles + 
            C.A.Conversation +
            C.A.PlotPoint +
            C.A.Scanner +
            C.A.Worksheet +
            C.A.WorksheetSubmit +
            Average,
          data=f.data)
summary(f.lm)

fd.lm = lm(NLG~scale(Duration) + scale(Average) + Condition, data=f.data)
summary(fd.lm)

gs.lm = lm(NLG~scale(FinalGameScore) +  scale(Average), data=f.data)
summary(gs.lm)

fgs.lm = lm(FinalGameScore~Average, data=f.data)
summary(fgs.lm)


# LOADING IN SLOPE DATAFRAME
s.data = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevisedSlope.csv")
cor.test(s.data[s.data[,"Condition"]==1, "Slope"], s.data[s.data[,"Condition"]==1, "NLG"])
cor.test(s.data[s.data[,"Condition"]==1, "Slope"], s.data[s.data[,"Condition"]==1, "Duration"])
hist(s.data[s.data[,"Condition"]==1,"Slope"])
plot(s.data[s.data[,"Condition"]==1,"Slope"], s.data[s.data[,"Condition"]==1, "NLG"])

cor.test(s.data[s.data[,"Condition"]==1, "CosSim"], s.data[s.data[,"Condition"]==1, "NLG"])
cor.test(s.data[s.data[,"Condition"]==1, "CosSim"], s.data[s.data[,"Condition"]==1, "Duration"])

cor(s.data[s.data[,"Condition"]==1,c("Average", "Slope", "CosSim")])

summary(lm(NLG~Average + CosSim,data=s.data[s.data[,"Condition"]==1,]))
