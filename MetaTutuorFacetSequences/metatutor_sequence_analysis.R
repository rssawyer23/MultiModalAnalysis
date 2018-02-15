## METATUTOR FACET SEQUENCES ANALYSIS

# Reading in student summary data
sd = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputSeptember/FACET-ThresholdCrossed/FACET-Sequence-Summary-Probabilities-Student.csv")

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

# Linear model predicting Multiple Choice Scores
mc_lm_s = lm(MultipleChoiceScores~ContentPage.Duration 
	+ FACET.Confusion 
	+ FACET.Frustration
	+ FACET.Neutral
	+ FACET.Joy,data=sd)
summary(mc_lm)

# Linear model using only the likelihood of relevant features (those with significant likelihoods
mc_lm_lt = lm(MultipleChoiceScores~L.FACET.Confusion.to.Frustration
	+ L.FACET.Frustration.to.Confusion,data=sd)
summary(mc_lm_lt)
plot(mc_lm_lt$fitted.values, mc_lm_lt$residuals)

# Linear model using same but as conditionals
mc_lm_t = lm(MultipleChoiceScores~FACET.Confusion.to.Frustration
	+ FACET.Frustration.to.Confusion,data=sd)
summary(mc_lm_t)
plot(mc_lm_t$fitted.values, mc_lm_t$residuals)

# Checking correlation of facet transitions to response differs using conditional or likelihood
cor.test(sd[,"L.FACET.Confusion.to.Frustration"],sd[,"MultipleChoiceScores"])
cor.test(sd[,"FACET.Confusion.to.Frustration"],sd[,"MultipleChoiceScores"])

cor.test(sd[,"L.FACET.Frustration.to.Confusion"],sd[,"MultipleChoiceScores"])
cor.test(sd[,"FACET.Frustration.to.Confusion"],sd[,"MultipleChoiceScores"])

