## METATUTOR FACET SEQUENCES ANALYSIS

# Reading in student summary data
sd = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Probabilities-Student.csv")
sd[is.na(sd)] = 0
sd = sd[sd[,"MultipleChoiceScores"] != -1,]
sd = sd[sd[,"TestSubject"] != "IVH2_PN050",]

hist(sd[,"FACET.Confusion"], breaks=15)
num_emotions = 7
# Relative proportions of tracked emotions, should report mean and std of proportions
apply(sd[,c("FACET.Confusion", "FACET.Frustration", "FACET.Fear", "FACET.Neutral", "FACET.Surprise", "FACET.Contempt", "FACET.Joy")], 2, mean)
apply(sd[,c("FACET.Confusion", "FACET.Frustration", "FACET.Fear", "FACET.Neutral", "FACET.Surprise", "FACET.Contempt", "FACET.Joy")], 2, sd)
t.test(sd[,"FACET.Confusion"] - 1 / num_emotions)
t.test(sd[,"FACET.Frustration"] - 1 / num_emotions)
t.test(sd[,"FACET.Fear"] - 1 / num_emotions)
t.test(sd[,"FACET.Neutral"] - 1 / num_emotions)
t.test(sd[,"FACET.Surprise"] - 1 / num_emotions)
t.test(sd[,"FACET.Contempt"] - 1 / num_emotions)
t.test(sd[,"FACET.Joy"] - 1 / num_emotions)

summary(sd[,c("FACET.Confusion.Freq", "FACET.Frustration.Freq", "FACET.Fear.Freq", "FACET.Neutral.Freq", "FACET.Surprise.Freq", "FACET.Contempt.Freq", "FACET.Joy.Freq", "ContentPage.Duration")])
apply(sd[,c("FACET.Confusion.Freq", "FACET.Frustration.Freq", "FACET.Fear.Freq", "FACET.Neutral.Freq", "FACET.Surprise.Freq", "FACET.Contempt.Freq", "FACET.Joy.Freq", "ContentPage.Duration")], 2, mean)
apply(sd[,c("FACET.Confusion.Freq", "FACET.Frustration.Freq", "FACET.Fear.Freq", "FACET.Neutral.Freq", "FACET.Surprise.Freq", "FACET.Contempt.Freq", "FACET.Joy.Freq", "ContentPage.Duration")], 2, sd)
apply(sd[,c("FACET.Confusion.Freq", "FACET.Frustration.Freq", "FACET.Fear.Freq", "FACET.Neutral.Freq", "FACET.Surprise.Freq", "FACET.Contempt.Freq", "FACET.Joy.Freq", "ContentPage.Duration")], 2, median)

apply(sd[,c("L.FACET.Confusion.to.Frustration", "L.FACET.Frustration.to.Confusion", "L.FACET.Confusion.to.Joy", "L.FACET.Frustration.to.Joy", "ContentPage.Duration")], 2, mean)
apply(sd[,c("L.FACET.Confusion.to.Frustration", "L.FACET.Frustration.to.Confusion", "L.FACET.Confusion.to.Joy", "L.FACET.Frustration.to.Joy", "ContentPage.Duration")], 2, sd)
apply(sd[,c("L.FACET.Confusion.to.Frustration", "L.FACET.Frustration.to.Confusion", "L.FACET.Confusion.to.Joy", "L.FACET.Frustration.to.Joy", "ContentPage.Duration")], 2, median)
t.test(sd[,"L.FACET.Confusion.to.Frustration"], sd[,"L.FACET.Frustration.to.Confusion"])
t.test(sd[,"L.FACET.Confusion.to.Frustration"])
cor(sd[,c("MultipleChoiceScores", "MultipleChoiceConfidence", "L.FACET.Confusion.to.Frustration", "L.FACET.Frustration.to.Confusion", "L.FACET.Confusion.to.Joy", "L.FACET.Frustration.to.Joy", "ContentPage.Duration")])
cor.test(sd[,"MultipleChoiceScores"], sd[,"L.FACET.Frustration.to.Confusion"])

cor(sd[,c("MultipleChoiceScores", "MultipleChoiceConfidence","FACET.Confusion.Freq", "FACET.Frustration.Freq", "FACET.Fear.Freq", "FACET.Neutral.Freq", "FACET.Surprise.Freq", "FACET.Contempt.Freq", "FACET.Joy.Freq", "ContentPage.Duration")])
hist(sd[,"FACET.Frustration.Freq"], breaks=10)
cor.test(sd[,"MultipleChoiceScores"], sd[,"FACET.Contempt.Freq"])

# Showing that emotions are not uniformly distributed
row_events = sd[,c("FACET.Confusion", "FACET.Frustration", "FACET.Fear", "FACET.Neutral", "FACET.Surprise", "FACET.Contempt", "FACET.Joy")]*sd[,"Length"]
events = apply(row_events,2, sum)
N = sum(row_events)
theo_events = rep(N/num_emotions,times=length(events))
chi_sq = sum((events - theo_events)^2 / theo_events); chi_sq
p_val = pchisq(chi_sq, df=num_emotions-1, lower.tail=FALSE); p_val
rs1 = chisq.test(events); rs1

# Showing that the number of affect events observed correlated to duration
# But the affect per second is not, and is normally distributed
cor.test(sd[,"Length"],sd[,"ContentPage.Duration"])
cor.test(sd[,"AffectPerSecond"],sd[,"ContentPage.Duration"])
hist(sd[,"Length"],breaks=16)
hist(sd[,"AffectPerSecond"],breaks=16)
shapiro.test(sd[,"Length"])
shapiro.test(sd[,"AffectPerSecond"])
plot(sd[,"Length"],sd[,"ContentPage.Duration"])
plot(sd[,"AffectPerSecond"],sd[,"ContentPage.Duration"])

# Running single sample t-tests on the D'Mello Likelihood measure (different from 0)
#  as done in the 2012 paper
t.test(sd[,"L.FACET.Confusion.to.Frustration"])
t.test(sd[,"L.FACET.Confusion.to.Joy"])
t.test(sd[,"L.FACET.Frustration.to.Joy"])
t.test(sd[,"L.FACET.Frustration.to.Confusion"])
t.test(sd[,"L.FACET.Confusion.to.Frustration"], sd[,"L.FACET.Frustration.to.Confusion"])


# Linear model predicting Multiple Choice Scores
cor(sd[,c("FACET.Confusion", "FACET.Frustration", "FACET.Fear", "FACET.Neutral", "FACET.Surprise", "FACET.Contempt", "FACET.Joy")])
mc_lm_s = lm(MultipleChoiceScores~ContentPage.Duration 
	+ FACET.Confusion.Freq
	+ FACET.Frustration.Freq
	+ FACET.Neutral.Freq
	+ FACET.Surprise.Freq
	+ FACET.Contempt.Freq
	+ FACET.Joy.Freq,data=sd)
summary(mc_lm_s)

# Linear model predicting Multiple Choice Confidence
mcc_lm_s = lm(MultipleChoiceConfidence~ContentPage.Duration 
             + FACET.Confusion.Freq 
             + FACET.Frustration.Freq
             + FACET.Joy.Freq,data=sd)
summary(mcc_lm_s)

jcc_lm_s = lm(JustificationConfidence~ContentPage.Duration 
              + FACET.Confusion.Freq 
              + FACET.Frustration.Freq
              + FACET.Joy.Freq,data=sd)
summary(jcc_lm_s)


cor.test(sd[,"L.FACET.Frustration.to.Confusion"], sd[,"L.FACET.Confusion.to.Frustration"])
# Linear model using only the likelihood of relevant features (those with significant likelihoods)
mc_lm_lt = lm(MultipleChoiceScores~L.FACET.Confusion.to.Frustration
	+ L.FACET.Frustration.to.Confusion,data=sd)
summary(mc_lm_lt)
plot(mc_lm_lt$fitted.values, mc_lm_lt$residuals)

# Linear model using same but as conditionals
mc_lm_t = lm(MultipleChoiceScores~FACET.Confusion.to.Frustration
	+ FACET.Frustration.to.Confusion,data=sd)
summary(mc_lm_t)
plot(mc_lm_t$fitted.values, mc_lm_t$residuals)

# Linear model using only the likelihood of relevant features for MC Confidence
mcc_lm_lt = lm(MultipleChoiceConfidence~L.FACET.Confusion.to.Frustration
               + L.FACET.Frustration.to.Confusion, data=sd)
summary(mcc_lm_lt)

# Checking correlation of facet transitions to response differs using conditional or likelihood
cor.test(sd[,"L.FACET.Confusion.to.Frustration"],sd[,"MultipleChoiceScores"])
cor.test(sd[,"FACET.Confusion.to.Frustration"],sd[,"MultipleChoiceScores"])

cor.test(sd[,"L.FACET.Frustration.to.Confusion"],sd[,"MultipleChoiceScores"])
cor.test(sd[,"FACET.Frustration.to.Confusion"],sd[,"MultipleChoiceScores"])

