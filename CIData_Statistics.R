#STD FILE DIVIDES BY DURATION IN INTERVAL FOR COUNT = RATE, DURATION = PROPORTION (e.g. All-FrustrationEvidence-Count is the count/full duration)
library(car)
data = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryStd.csv")
data = data[ !(data$TestSubject %in% c("CI1302PN116", "CI1302PN011", "CI1301PN042", "CI1301PN043", "CI1302PN098")),]
cond = data[,"Condition"]
data[cond=="3","Duration"] = rep(92*60,sum(cond=="3"))
data["NLGd"] = data[,"NLG"] / data[,"Duration"]
cor.test(data[,"NLG"],data[,"Duration"])
table(cond)

# SOME DESCRIPTIVE STATISTICS FOR INCLUSION
unstd_data = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGradedActions.csv")
unstd_data = unstd_data[ !(unstd_data$TestSubject %in% c("CI1302PN116", "CI1302PN011", "CI1301PN042", "CI1301PN043", "CI1302PN098")),]
unstd_data = unstd_data[unstd_data[,"Condition"]!=3,]
dim(unstd_data) # Should have 99 rows here
names(unstd_data)
apply(unstd_data[unstd_data[,"Condition"]==1,c("Duration.PreScan","Duration.PostScan","Duration", "PostScan.Movement.Duration")]/60, 2, mean)
apply(unstd_data[unstd_data[,"Condition"]==1,c("Duration.PreScan","Duration.PostScan","Duration", "PostScan.Movement.Duration")]/60, 2, sd)
apply(unstd_data[unstd_data[,"Condition"]==2,c("Duration.PreScan","Duration.PostScan","Duration", "PostScan.Movement.Duration")]/60, 2, mean)
apply(unstd_data[unstd_data[,"Condition"]==2,c("Duration.PreScan","Duration.PostScan","Duration", "PostScan.Movement.Duration")]/60, 2, sd)


## 1.1 ANCOVA FOR NLG BY CONDITION CONTROLLING FOR DURATION
lm_nlg_cond  = lm(NLG~Duration + factor(Condition),data=data)
lm_nlg_condc = lm(NLG~Duration * factor(Condition), data=data)
lm_nlgd_cond = lm(NLGd ~ factor(Condition),data=data)
summary(lm_nlg_cond)
Anova(lm_nlg_cond, type="III")
summary(lm_nlg_condc)
anova(lm_nlg_condc)
summary(lm_nlgd_cond)
anova(lm_nlgd_cond)

# 1.2 MANCOVA FOR SURVEY MEASURES BY CONDITION
attach(data)
Y = cbind(Post.Presence.Involvement, Post.Presence.Sensory.Fidelity, Post.Presence.Adaptation.Immersion,Post.Presence.Interface.Quality)
Y = cbind(Y, Post.IMI.Interest.Enjoyment,Post.IMI.Perceived.Competence,Post.IMI.Effort.Importance,Post.IMI.Pressure.Tension,Post.IMI.Value.Usefulness)
fit = manova(Y[data[,"Condition"]!=3,]~Duration * factor(Condition),data=data[data[,"Condition"]!=3,])
summary(fit,test="Pillai")
anova(fit)
summary.aov(fit)
f1 = aov(Post.Presence.Involvement~factor(Condition),data=data)
TukeyHSD(f1)
summary(lm(Post.IMI.Pressure.Tension~factor(Condition),data=data[data[,"Condition"]!=3,]))

# 1.3 MANCOVA FOR REPEATED MEASURES OF EV SURVEY
Y = cbind(Post.EV - Pre.EV, Post.EV.Negative - Pre.EV.Negative, Post.EV.Positive - Pre.EV.Positive)
fit = manova(Y[data[,"Condition"]!=3,]~Duration * factor(Condition),data=data[data[,"Condition"]!=3,])
apply(Y[data[,"Condition"]=="1",], 2, mean)
apply(Y[data[,"Condition"]=="1",], 2, sd)
apply(Y[data[,"Condition"]=="2",], 2, mean)
apply(Y[data[,"Condition"]=="2",], 2, sd)
summary(fit,test="Pillai")
summary.aov(fit)
summary(lm(Y[data[,"Condition"]!=3,]~ Duration * factor(Condition),data=data[data[,"Condition"]!="3",]))


# 1.4 MANCOVA FOR OVERALL FACET (FRUST, CONF, JOY)
Y = cbind(All.FrustrationEvidence.Duration, All.ConfusionEvidence.Duration, All.JoyEvidence.Duration)
fit = manova(Y[data[,"Condition"]!=3,]~Duration * factor(Condition),data=data[data[,"Condition"]!=3,])
summary(fit,test="Pillai")
summary.aov(fit)
summary(frust_test<-lm(All.FrustrationEvidence.Duration~Duration * factor(Condition),data=data[data[,"Condition"]!="3",]))
summary(lm(All.ConfusionEvidence.Duration~Duration * factor(Condition),data=data[data[,"Condition"]!="3",]))
summary(lm(All.JoyEvidence.Duration~Duration * factor(Condition),data=data[data[,"Condition"]!="3",]))




#######################################################
sdata = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryStd.csv")
sdata = sdata[ !(sdata$TestSubject %in% c("CI1302PN116", "CI1302PN011", "CI1301PN042", "CI1301PN043", "CI1302PN098")),]
## 2.1.a Agency and stage effect on rates of action types
attach(sdata)
# On full gameplay of full and partial
Yall = cbind(All.KnowledgeAcquisition.Count, All.InformationGathering.Count, All.HypothesisTesting.Count)
fit = manova(Yall[sdata[,"Condition"]!=3,]~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,])
summary(fit,test="Pillai")
summary.aov(fit)
Anova(fit, type="III")
# Counts are standardized by time, so the correlations are not obvious. For example, KA and IG are negative, but HT is positive
summary(lm(All.KnowledgeAcquisition.Count~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(All.InformationGathering.Count~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(All.HypothesisTesting.Count~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))

# On prescan phase of gameplay of full and partial
Ypre = cbind(PreScan.KnowledgeAcquisition.Count, PreScan.InformationGathering.Count, PreScan.HypothesisTesting.Count)
fit = manova(Ypre[sdata[,"Condition"]!=3,]~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,])
summary(fit,test="Pillai")
Anova(fit, type="III")
summary.aov(fit)
summary(lm(PreScan.KnowledgeAcquisition.Count~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PreScan.InformationGathering.Count~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PreScan.HypothesisTesting.Count~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))

# On postscan phase of gameplay of full and partial
Ypost = cbind(PostScan.KnowledgeAcquisition.Count, PostScan.InformationGathering.Count, PostScan.HypothesisTesting.Count)
fit = manova(Ypost[sdata[,"Condition"]!=3,]~Duration.PostScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,])
summary(fit,test="Pillai")
Anova(fit, type="III")
summary.aov(fit)

#############################
## 2.1.b Agency and stage effect on durations of action types
# On full gameplay of full and partial
Yall = cbind(All.KnowledgeAcquisition.Duration, All.InformationGathering.Duration, All.HypothesisTesting.Duration)
fit = manova(scale(Yall[sdata[,"Condition"]!=3,])~scale(Duration) * factor(Condition),data=sdata[sdata[,"Condition"]!=3,])
summary(fit,test="Pillai")
Anova(fit, type="III")
summary.aov(fit)
summary(lm(All.KnowledgeAcquisition.Duration~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(All.InformationGathering.Duration~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(All.HypothesisTesting.Duration~Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))

# On prescan phase of gameplay of full and partial
Ypre = cbind(PreScan.KnowledgeAcquisition.Duration, PreScan.InformationGathering.Duration, PreScan.HypothesisTesting.Duration)
fit = manova(scale(Ypre[sdata[,"Condition"]!=3,])~scale(Duration.PreScan) * factor(Condition),data=sdata[sdata[,"Condition"]!=3,])
summary(fit,test="Pillai")
Anova(fit, type="III")
summary.aov(fit)
summary(lm(PreScan.KnowledgeAcquisition.Duration~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PreScan.InformationGathering.Duration~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PreScan.HypothesisTesting.Duration~Duration.PreScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PreScan.BooksAndArticles.Count~Duration.PreScan*factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))


# On postscan phase of gameplay of full and partial
Ypost = cbind(PostScan.KnowledgeAcquisition.Duration, PostScan.InformationGathering.Duration, PostScan.HypothesisTesting.Duration)
fit = manova(scale(Ypost[sdata[,"Condition"]!=3,])~scale(Duration.PostScan) * factor(Condition),data=sdata[sdata[,"Condition"]!=3,])
summary(fit,test="Pillai")
Anova(fit, type="III")
summary.aov(fit)
summary(lm(PostScan.KnowledgeAcquisition.Duration~Duration.PostScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PostScan.InformationGathering.Duration~Duration.PostScan * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(scale(PostScan.HypothesisTesting.Duration)~scale(Duration.PostScan) * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))

summary(lm(PostScan.KnowledgeAcquisition.Duration~PreScan.KnowledgeAcquisition.Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PostScan.InformationGathering.Duration~PreScan.InformationGathering.Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))
summary(lm(PostScan.HypothesisTesting.Duration~PreScan.HypothesisTesting.Duration * factor(Condition),data=sdata[sdata[,"Condition"]!=3,]))

########################################################################
wdata = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsAllCategories.csv")
wdata.pre = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsPreScanCategories.csv")
wdata.post = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsPostScanCategories.csv")

wdata = wdata[ !(wdata$TestSubject %in% c("CI1302PN116", "CI1302PN011", "CI1301PN042", "CI1301PN043", "CI1302PN098")),]
wdata.pre = wdata.pre[ !(wdata.pre$TestSubject %in% c("CI1302PN116", "CI1302PN011", "CI1301PN042", "CI1301PN043", "CI1302PN098")),]
wdata.post = wdata.post[ !(wdata.post$TestSubject %in% c("CI1302PN116", "CI1302PN011", "CI1301PN042", "CI1301PN043", "CI1302PN098")),]

wdata["Condition"] = apply(wdata["TestSubject"], MARGIN=1, FUN=substr, start=6, stop=6)
wdata.pre["Condition"] = apply(wdata.pre["TestSubject"], MARGIN=1, FUN=substr, start=6, stop=6)
wdata.post["Condition"] = apply(wdata.post["TestSubject"], MARGIN=1, FUN=substr, start=6, stop=6)

## 2.2.a Agency, stage, window effect on contextualized emotion counts
# On full gameplay of full and partial agency conditions
Yall = cbind(wdata[,"During.KnowledgeAcquisition.JoyEvidence.CountRate"], 
	wdata[,"During.KnowledgeAcquisition.ConfusionEvidence.CountRate"], 
	wdata[,"During.KnowledgeAcquisition.FrustrationEvidence.CountRate"],
	wdata[,"During.InformationGathering.JoyEvidence.CountRate"], 
	wdata[,"During.InformationGathering.ConfusionEvidence.CountRate"], 
	wdata[,"During.InformationGathering.FrustrationEvidence.CountRate"],
	wdata[,"After.HypothesisTesting.JoyEvidence.CountRate"], 
	wdata[,"After.HypothesisTesting.ConfusionEvidence.CountRate"], 
	wdata[,"After.HypothesisTesting.FrustrationEvidence.CountRate"])
fit = manova(scale(Yall)~scale(wdata[,"Duration"]) * factor(wdata[,"Condition"]))
summary(fit,test="Pillai")
summary.aov(fit)

# On PreScan interval of full and partial agency conditions
Ypre = cbind(wdata.pre[,"During.KnowledgeAcquisition.JoyEvidence.CountRate"], 
	wdata.pre[,"During.KnowledgeAcquisition.ConfusionEvidence.CountRate"], 
	wdata.pre[,"During.KnowledgeAcquisition.FrustrationEvidence.CountRate"],
	wdata.pre[,"During.InformationGathering.JoyEvidence.CountRate"], 
	wdata.pre[,"During.InformationGathering.ConfusionEvidence.CountRate"], 
	wdata.pre[,"During.InformationGathering.FrustrationEvidence.CountRate"],
	wdata.pre[,"After.HypothesisTesting.JoyEvidence.CountRate"], 
	wdata.pre[,"After.HypothesisTesting.ConfusionEvidence.CountRate"], 
	wdata.pre[,"After.HypothesisTesting.FrustrationEvidence.CountRate"])
fit = manova(scale(Ypre)~scale(wdata.pre[,"Duration.PreScan"]) * factor(wdata.pre[,"Condition"]))
summary(fit,test="Pillai")
summary.aov(fit)

# On PostScan interval of full and partial agency conditions
Ypost = cbind(wdata.post[,"During.KnowledgeAcquisition.JoyEvidence.CountRate"], 
	wdata.post[,"During.KnowledgeAcquisition.ConfusionEvidence.CountRate"], 
	wdata.post[,"During.KnowledgeAcquisition.FrustrationEvidence.CountRate"],
	wdata.post[,"During.InformationGathering.JoyEvidence.CountRate"], 
	wdata.post[,"During.InformationGathering.ConfusionEvidence.CountRate"], 
	wdata.post[,"During.InformationGathering.FrustrationEvidence.CountRate"],
	wdata.post[,"After.HypothesisTesting.JoyEvidence.CountRate"], 
	wdata.post[,"After.HypothesisTesting.ConfusionEvidence.CountRate"], 
	wdata.post[,"After.HypothesisTesting.FrustrationEvidence.CountRate"])
fit = manova(Ypost~wdata.post[,"Duration.PostScan"] * factor(wdata.post[,"Condition"]))
summary(fit,test="Pillai")
summary.aov(fit)

## 2.2.b Agency, stage, window effect on contextualized emotion durations
# On full gameplay of full and partial agency conditions
Yall = cbind(wdata[,"During.KnowledgeAcquisition.JoyEvidence.DurationProp"], 
	wdata[,"During.KnowledgeAcquisition.ConfusionEvidence.DurationProp"], 
	wdata[,"During.KnowledgeAcquisition.FrustrationEvidence.DurationProp"],
	wdata[,"During.InformationGathering.JoyEvidence.DurationProp"], 
	wdata[,"During.InformationGathering.ConfusionEvidence.DurationProp"], 
	wdata[,"During.InformationGathering.FrustrationEvidence.DurationProp"],
	wdata[,"After.HypothesisTesting.JoyEvidence.DurationProp"], 
	wdata[,"After.HypothesisTesting.ConfusionEvidence.DurationProp"], 
	wdata[,"After.HypothesisTesting.FrustrationEvidence.DurationProp"])
fit = manova(scale(Yall)~scale(wdata[,"Duration"]) * factor(wdata[,"Condition"]))
Anova(fit, type="III")
summary(lm(scale(Yall)~scale(wdata[,"Duration"]) * factor(wdata[,"Condition"])))

summary(fit,test="Pillai")
summary.aov(fit)


# On PreScan interval of full and partial agency conditions
Ypre = cbind(wdata.pre[,"During.KnowledgeAcquisition.JoyEvidence.DurationProp"], 
	wdata.pre[,"During.KnowledgeAcquisition.ConfusionEvidence.DurationProp"], 
	wdata.pre[,"During.KnowledgeAcquisition.FrustrationEvidence.DurationProp"],
	wdata.pre[,"During.InformationGathering.JoyEvidence.DurationProp"], 
	wdata.pre[,"During.InformationGathering.ConfusionEvidence.DurationProp"], 
	wdata.pre[,"During.InformationGathering.FrustrationEvidence.DurationProp"],
	wdata.pre[,"After.HypothesisTesting.JoyEvidence.DurationProp"], 
	wdata.pre[,"After.HypothesisTesting.ConfusionEvidence.DurationProp"], 
	wdata.pre[,"After.HypothesisTesting.FrustrationEvidence.DurationProp"])
fit = manova(scale(Ypre)~scale(wdata.pre[,"Duration.PreScan"]) * factor(wdata.pre[,"Condition"]))
Anova(fit, type="III")
summary(fit,test="Pillai")
summary.aov(fit)

# On PostScan interval of full and partial agency conditions
Ypost = cbind(wdata.post[,"During.KnowledgeAcquisition.JoyEvidence.DurationProp"], 
	wdata.post[,"During.KnowledgeAcquisition.ConfusionEvidence.DurationProp"], 
	wdata.post[,"During.KnowledgeAcquisition.FrustrationEvidence.DurationProp"],
	wdata.post[,"During.InformationGathering.JoyEvidence.DurationProp"], 
	wdata.post[,"During.InformationGathering.ConfusionEvidence.DurationProp"], 
	wdata.post[,"During.InformationGathering.FrustrationEvidence.DurationProp"],
	wdata.post[,"After.HypothesisTesting.JoyEvidence.DurationProp"], 
	wdata.post[,"After.HypothesisTesting.ConfusionEvidence.DurationProp"], 
	wdata.post[,"After.HypothesisTesting.FrustrationEvidence.DurationProp"])
fit = manova(scale(Ypost)~scale(wdata.post[,"Duration.PostScan"]) * factor(wdata.post[,"Condition"]))
summary(lm(scale(Ypost)~scale(wdata.post[,"Duration.PostScan"]) * factor(wdata.post[,"Condition"])))
summary(fit,test="Pillai")
Anova(fit, type="III")
summary.aov(fit)
summary(lm(During.KnowledgeAcquisition.JoyEvidence.DurationProp~Duration.PostScan*factor(Condition),data=wdata.post))

#################################################################################################
# 3 Linear Model of NLG
# Checking for normality of response variables
sdata = data[data[,"Condition"]!="3",]
sdata = sdata[sdata[,"TestSubject"]!="CI1301PN008",]
hist(sdata[,"NLG"],breaks=10)
shapiro.test(sdata[,"NLG"])
qqnorm(sdata[,"NLG"])
qqline(sdata[,"NLG"])

cor(sdata[,c("Pre.EV.Negative","Pre.EV.Positive")])
cor(sdata[,c("Pre.AGQ.Mastery.Approach","Pre.AGQ.Performance.Approach","Pre.AGQ.Mastery.Avoidance","Pre.AGQ.Performance.Avoidance")])

wdata = wdata[wdata["TestSubject"]!="CI1301PN008",]

alldata = cbind(wdata,sdata)
sum(factor(alldata[,1]) != factor(alldata[,807]))  # THIS SHOULD BE 0 IF CONCATENATION DONE CORRECTLY



# Models without Interaction term but including condition
lm_null = lm(NLG~1,data=alldata)

lm_base = lm(NLG~factor(Condition)
	 + Duration, data=alldata)

lm_acts = lm(NLG~factor(Condition) 
	+ Duration 
	+ All.KnowledgeAcquisition.Duration 
	+ All.InformationGathering.Duration 
	+ All.HypothesisTesting.Duration,data=alldata)

lm_acts_scales = lm(NLG~factor(Condition) 
	+ Duration 
	+ All.KnowledgeAcquisition.Duration 
	+ All.InformationGathering.Duration 
	+ All.HypothesisTesting.Duration
	+ Pre.EV.Negative
	+ Pre.EV.Positive
	+ Pre.AGQ.Mastery.Approach
	+ Pre.AGQ.Mastery.Avoidance
	+ Pre.AGQ.Performance.Approach
	+ Pre.AGQ.Performance.Avoidance,data=alldata)

lm_acts_surveys = lm(NLG~factor(Condition) 
	+ Duration 
	+ All.KnowledgeAcquisition.Duration 
	+ All.InformationGathering.Duration 
	+ All.HypothesisTesting.Duration
	+ Pre.EV
	+ Pre.AGQ,data=alldata)

lm_acts_context = lm(NLG~factor(Condition) 
	+ Duration 
	+ All.KnowledgeAcquisition.Duration 
	+ All.InformationGathering.Duration 
	+ All.HypothesisTesting.Duration
	+ During.KnowledgeAcquisition.JoyEvidence.DurationProp 
	+ During.KnowledgeAcquisition.ConfusionEvidence.DurationProp
	+ During.KnowledgeAcquisition.FrustrationEvidence.DurationProp
	+ During.InformationGathering.JoyEvidence.DurationProp
	+ During.InformationGathering.ConfusionEvidence.DurationProp
	+ During.InformationGathering.FrustrationEvidence.DurationProp
	+ After.HypothesisTesting.JoyEvidence.DurationProp
	+ After.HypothesisTesting.ConfusionEvidence.DurationProp
	+ After.HypothesisTesting.FrustrationEvidence.DurationProp ,data=alldata)

lm_all = lm(NLG~factor(Condition) 
	+ Duration 
	+ All.KnowledgeAcquisition.Duration 
	+ All.InformationGathering.Duration 
	+ All.HypothesisTesting.Duration
	+ During.KnowledgeAcquisition.JoyEvidence.DurationProp 
	+ During.KnowledgeAcquisition.ConfusionEvidence.DurationProp
	+ During.KnowledgeAcquisition.FrustrationEvidence.DurationProp
	+ During.InformationGathering.JoyEvidence.DurationProp
	+ During.InformationGathering.ConfusionEvidence.DurationProp
	+ During.InformationGathering.FrustrationEvidence.DurationProp
	+ After.HypothesisTesting.JoyEvidence.DurationProp
	+ After.HypothesisTesting.ConfusionEvidence.DurationProp
	+ After.HypothesisTesting.FrustrationEvidence.DurationProp 
	+ Pre.EV.Negative
	+ Pre.EV.Positive
	+ Pre.AGQ.Mastery.Approach
	+ Pre.AGQ.Mastery.Avoidance
	+ Pre.AGQ.Performance.Approach
	+ Pre.AGQ.Performance.Avoidance,data=alldata)

step(lm_null, scope=list(upper=lm_all), data=alldata, direction="both")
lm_step = lm(scale(NLG)~factor(Condition)
	+ scale(All.HypothesisTesting.Duration)
	+ scale(Pre.AGQ.Mastery.Avoidance)
	+ scale(During.KnowledgeAcquisition.FrustrationEvidence.DurationProp),data=alldata)
summary(lm_step)

## Same analysis with interaction terms
lm_base_i = lm(NLG~factor(Condition) * Duration, data=alldata)

lm_acts_i = lm(NLG~factor(Condition) * Duration 
	+ All.KnowledgeAcquisition.Duration * factor(Condition)
	+ All.InformationGathering.Duration * factor(Condition)
	+ All.HypothesisTesting.Duration * factor(Condition),data=alldata)

lm_acts_scales_i = lm(NLG~ Duration * factor(Condition) 
	+ All.KnowledgeAcquisition.Duration * factor(Condition) 
	+ All.InformationGathering.Duration * factor(Condition) 
	+ All.HypothesisTesting.Duration* factor(Condition) 
	+ Pre.EV.Negative* factor(Condition) 
	+ Pre.EV.Positive* factor(Condition) 
	+ Pre.AGQ.Mastery.Approach* factor(Condition) 
	+ Pre.AGQ.Mastery.Avoidance* factor(Condition) 
	+ Pre.AGQ.Performance.Approach* factor(Condition) 
	+ Pre.AGQ.Performance.Avoidance* factor(Condition) ,data=alldata)

lm_acts_surveys_i = lm(NLG~factor(Condition)* Duration 
	+ All.KnowledgeAcquisition.Duration * factor(Condition)
	+ All.InformationGathering.Duration * factor(Condition)
	+ All.HypothesisTesting.Duration* factor(Condition)
	+ Pre.EV* factor(Condition)
	+ Pre.AGQ* factor(Condition),data=alldata)

lm_acts_context_i = lm(NLG~factor(Condition) * Duration 
	+ All.KnowledgeAcquisition.Duration * factor(Condition)
	+ All.InformationGathering.Duration * factor(Condition)
	+ All.HypothesisTesting.Duration* factor(Condition)
	+ During.KnowledgeAcquisition.JoyEvidence.DurationProp * factor(Condition)
	+ During.KnowledgeAcquisition.ConfusionEvidence.DurationProp* factor(Condition)
	+ During.KnowledgeAcquisition.FrustrationEvidence.DurationProp* factor(Condition)
	+ During.InformationGathering.JoyEvidence.DurationProp* factor(Condition)
	+ During.InformationGathering.ConfusionEvidence.DurationProp* factor(Condition)
	+ During.InformationGathering.FrustrationEvidence.DurationProp* factor(Condition)
	+ After.HypothesisTesting.JoyEvidence.DurationProp* factor(Condition)
	+ After.HypothesisTesting.ConfusionEvidence.DurationProp* factor(Condition)
	+ After.HypothesisTesting.FrustrationEvidence.DurationProp* factor(Condition) ,data=alldata)

lm_all_i = lm(NLG~factor(Condition) * Duration 
	+ All.KnowledgeAcquisition.Duration * factor(Condition)
	+ All.InformationGathering.Duration * factor(Condition)
	+ All.HypothesisTesting.Duration* factor(Condition)
	+ During.KnowledgeAcquisition.JoyEvidence.DurationProp * factor(Condition)
	+ During.KnowledgeAcquisition.ConfusionEvidence.DurationProp* factor(Condition)
	+ During.KnowledgeAcquisition.FrustrationEvidence.DurationProp* factor(Condition)
	+ During.InformationGathering.JoyEvidence.DurationProp* factor(Condition)
	+ During.InformationGathering.ConfusionEvidence.DurationProp* factor(Condition)
	+ During.InformationGathering.FrustrationEvidence.DurationProp* factor(Condition)
	+ After.HypothesisTesting.JoyEvidence.DurationProp* factor(Condition)
	+ After.HypothesisTesting.ConfusionEvidence.DurationProp* factor(Condition)
	+ After.HypothesisTesting.FrustrationEvidence.DurationProp * factor(Condition)
	+ Pre.EV.Negative* factor(Condition)
	+ Pre.EV.Positive* factor(Condition)
	+ Pre.AGQ.Mastery.Approach* factor(Condition)
	+ Pre.AGQ.Mastery.Avoidance* factor(Condition)
	+ Pre.AGQ.Performance.Approach* factor(Condition)
	+ Pre.AGQ.Performance.Avoidance* factor(Condition),data=alldata)

step(lm_null, scope=list(upper=lm_all_i), data=alldata, direction="both")
lm_step_i = lm(scale(NLG)~factor(Condition) * scale(All.HypothesisTesting.Duration)
	+ scale(Pre.AGQ.Mastery.Avoidance)
	+ scale(During.KnowledgeAcquisition.FrustrationEvidence.DurationProp),data=alldata)
summary(lm_step_i)
540/18














