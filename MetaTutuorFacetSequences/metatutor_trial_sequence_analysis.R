
# Reading in student-trial level data (18 * number students)
td = read.csv("C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Probabilities-Trial.csv")
td[is.na(td)] = 0  # Fixing rows with undefined (unseen) values
td = td[td[,"MultipleChoiceScores"] != -1,]  # Fixing ungraded MC Response
td = td[td[,"TestSubject"] != "IVH2_PN050",]

hist(td[,"FACET.Confusion"], breaks=20)
num_emotions = 7

# Showing that emotions are not uniformly distributed
row_events = td[,c("FACET.Confusion", "FACET.Frustration", "FACET.Fear", "FACET.Neutral", "FACET.Surprise", "FACET.Contempt", "FACET.Joy")]*td[,"Length"]
events = apply(row_events,2, sum); events
N = sum(row_events)
theo_events = rep(N/num_emotions,times=length(events))
chi_sq = sum((events - theo_events)^2 / theo_events); chi_sq
p_val = pchisq(chi_sq, df=num_emotions-1, lower.tail=FALSE); p_val
rs1 = chisq.test(events); rs1

t.test(td[,"L.FACET.Confusion.to.Frustration"])
t.test(td[,"L.FACET.Confusion.to.Joy"])
t.test(td[,"L.FACET.Frustration.to.Joy"])
t.test(td[,"L.FACET.Frustration.to.Confusion"])
t.test(td[,"L.FACET.Confusion.to.Frustration"], td[,"L.FACET.Frustration.to.Confusion"])

cor.test(td[,"FACET.Confusion"], td[,"FACET.Frustration"])
# Linear model predicting Multiple Choice Scores
mc_lm_s = glm(MultipleChoiceScores~
             FACET.Confusion.Freq
             + FACET.Frustration.Freq
             + FACET.Joy.Freq
             + L.FACET.Confusion.to.Frustration,data=td, family=binomial)
summary(mc_lm_s)
predictions = predict(mc_lm_s,type='response')
table(td[,"MultipleChoiceScores"], predictions > 0.5)

Y = cbind(td[,"FACET.Frustration.Freq"], td[,"FACET.Confusion.Freq"], td[,"FACET.Joy.Freq"], td[,"L.FACET.Confusion.to.Frustration"])
fit = manova(Y~td[,"MultipleChoiceScores"])
summary(fit)
summary.aov(fit)



mc_lm_s = lm(MultipleChoiceScores~
              factor(FACET.Confusion.Freq > 0)
              * factor(FACET.Frustration.Freq > 0)
              * factor(FACET.Joy.Freq > 0)
              + factor(L.FACET.Confusion.to.Frustration > 0),data=td)
summary(mc_lm_s)
anova(mc_lm_s)

mc_lm_s.full = glmer(MultipleChoiceScores~
                 factor(FACET.Confusion.Freq > 0)
               + factor(FACET.Frustration.Freq > 0)
               + factor(FACET.Joy.Freq > 0)
               + factor(L.FACET.Confusion.to.Frustration > 0)
               + (1|TestSubject), data=td, family=binomial)
summary(mc_lm_s.full)
mc_lm_s.null = glmer(MultipleChoiceScores~(1|TestSubject), data=td, family=binomial)
summary(mc_lm_s.null)
anova(mc_lm_s.null,mc_lm_s.full)

# CHI SQUARE SHOWING NO RELATIONSHIP WITH MC SCORES
c = td[,"FACET.Confusion.Freq"] > 0
f = td[,"FACET.Frustration.Freq"] > 0
j = td[,"FACET.Joy.Freq"] > 0
t = td[,"L.FACET.Confusion.to.Frustration"] > 0


predictions = predict(mc_lm_s,type='response')
table(td[,"MultipleChoiceScores"], predictions > 0.5)

# Model of multiple choice confidence
mcc_lm_s = lm(MultipleChoiceConfidence~ContentPage.Duration 
             + FACET.Confusion
             + FACET.Frustration
             + FACET.Neutral
             + FACET.Fear
             + FACET.Surprise
             + FACET.Contempt
             + FACET.Joy,data=td)
summary(mcc_lm_s)

cor.test(td[,"FACET.Confusion.Freq"], td[,"MultipleChoiceScores"])
cor.test(td[,"FACET.Frustration.Freq"], td[,"MultipleChoiceScores"])

mc_lm_lt = lm(MultipleChoiceScores~L.FACET.Confusion.to.Frustration
              + L.FACET.Frustration.to.Confusion,data=td)
summary(mc_lm_lt)

mcc_lm_lt = lm(MultipleChoiceConfidence~L.FACET.Confusion.to.Frustration
               + L.FACET.Frustration.to.Confusion,data=td)
summary(mcc_lm_lt)

# Testing differences in correct/incorrect multiple choice questions
correct = td[,"MultipleChoiceScores"] == 1
t.test(td[correct, "FACET.Confusion.Freq"], td[correct == FALSE, "FACET.Confusion.Freq"])
t.test(td[correct, "FACET.Frustration.Freq"], td[correct == FALSE, "FACET.Frustration.Freq"])

# Experienced Confusion
conf = td[,"FACET.Confusion"] != 0
frust = td[,"FACET.Frustration"] != 0
joy = td[,"FACET.Joy"] != 0
c.to.f = td[,"FACET.Confusion.to.Frustration"] != 0
t.test(td[conf,"MultipleChoiceScores"], td[conf==FALSE,"MultipleChoiceScores"])
t.test(td[frust,"MultipleChoiceScores"],td[frust==FALSE,"MultipleChoiceScores"])
t.test(td[c.to.f,"MultipleChoiceScores"], td[c.to.f==FALSE,"MultipleChoiceScores"])
t.test(td[joy,"MultipleChoiceScores"], td[joy==FALSE,"MultipleChoiceScores"])

t.test(td[conf,"MultipleChoiceConfidence"], td[conf==FALSE,"MultipleChoiceConfidence"])
t.test(td[frust,"MultipleChoiceConfidence"],td[frust==FALSE,"MultipleChoiceConfidence"])
t.test(td[c.to.f,"MultipleChoiceConfidence"], td[c.to.f==FALSE,"MultipleChoiceConfidence"])
t.test(td[joy, "MultipleChoiceConfidence"], td[joy==FALSE,"MultipleChoiceConfidence"])
head(td)

# Determining if manipulations have an effect on Confusion rate
c_lm = lm(FACET.Confusion.Freq~factor(AgentCongruence) + factor(TextRelevancy) + factor(DiagramRelevancy), data=td)
summary(c_lm)

c_mm = lmer(FACET.Confusion.Freq~factor(AgentCongruence) + factor(TextRelevancy) + factor(DiagramRelevancy) +
              (1 | TestSubject), data=td)
summary(c_mm)

f_mm = lmer(FACET.Frustration.Freq~factor(AgentCongruence)+ factor(TextRelevancy) + factor(DiagramRelevancy) +
              (1 | TestSubject), data=td, REML=FALSE)
f_mm.null = lmer(FACET.Frustration.Freq~(1 | TestSubject), data=td, REML=FALSE)
summary(f_mm)
summary(f_mm.null)
anova(f_mm.null,f_mm)

f_lm = lm(FACET.Frustration~factor(AgentCongruence) + factor(TextRelevancy) + factor(DiagramRelevancy), data=td)
summary(f_lm)

# Mixed effect model for MC Confidence using affect as fixed effect, Subject as random effect
mc_tss = sum((td[,"MultipleChoiceConfidence"] - mean(td[,"MultipleChoiceConfidence"]))**2)
mcc_mm = lmer(MultipleChoiceConfidence~factor(FACET.Confusion.Freq > 0) +
                factor(FACET.Frustration.Freq > 0) + 
                factor(FACET.Joy.Freq > 0) + 
                factor(L.FACET.Confusion.to.Frustration > 0) +
                (1 | TestSubject), data=td, REML=FALSE)
summary(mcc_mm)
mcc_mm.rss = sum((predict(mcc_mm) - td[,"MultipleChoiceConfidence"])**2)
mcc_mm.r2 = 1 - mcc_mm.rss/mc_tss; mcc_mm.r2

mcc_mm.ff = lmer(MultipleChoiceConfidence~L.FACET.Confusion.to.Frustration + 
                FACET.Joy.Freq + 
                (1 | TestSubject), data=td, REML=FALSE)
summary(mcc_mm.ff)
mcc_mm.ff.rss = sum((predict(mcc_mm.ff) - td[,"MultipleChoiceConfidence"])**2)
mcc_mm.ff.r2 = 1 - mcc_mm.ff.rss/mc_tss; mcc_mm.ff.r2

mcc_mm.null = lm(MultipleChoiceConfidence~1, data=td)
mcc_mm.null.rss = sum((predict(mcc_mm.null) - td[,"MultipleChoiceConfidence"])**2)
mcc_mm.null.r2 = 1 - mcc_mm.null.rss/mc_tss; mcc_mm.null.r2
summary(mcc_mm.null)

mcc_mm.re = lmer(MultipleChoiceConfidence~(1 | TestSubject), data=td, REML=FALSE)
mcc_mm.re.rss = sum((predict(mcc_mm.re) - td[,"MultipleChoiceConfidence"])**2)
mcc_mm.re.r2 = 1 - mcc_mm.re.rss/mc_tss; mcc_mm.re.r2

# F test of signficance for mixed effects models
n = dim(td)[1]
Fnum = (mcc_mm.re.rss - mcc_mm.ff.rss) / (n-55 - (n-57))
Fden = Fd = mcc_mm.ff.rss / (n-57)
Fstat = Fnum / Fden
pval = 1-pf(Fstat, df1=57-55, df2=n-55)
list(Fstat=Fstat, df1=57-55, df2=n-55, p=pval)

# F test of significance of mixed effect model with more parameters
full_params = 59
reduced_params = 55
Fnum = (mcc_mm.re.rss - mcc_mm.rss) / (n-reduced_params - (n-full_params))
Fden = Fd = mcc_mm.rss / (n-full_params)
Fstat = Fnum / Fden
pval = 1-pf(Fstat, df1=full_params-reduced_params, df2=n-reduced_params)
list(Fstat=Fstat, df1=full_params-reduced_params, df2=n-reduced_params, p=pval, r2=mcc_mm.r2)

# Chi square test of independence for likelihood ratio test of mixed effects models
anova(mcc_mm.re,mcc_mm)
anova(mcc_mm.re,mcc_mm.ff)

mcc_mm.test = lmer(MultipleChoiceConfidence~(1+factor(FACET.Confusion > 0)+factor(FACET.Frustration > 0)+factor(FACET.Joy>0)|TestSubject),data=td)
summary(mcc_mm.test)
mcc_mm.test

mcc_lm = lm(MultipleChoiceConfidence~factor(FACET.Confusion>0)
            +factor(FACET.Frustration>0)
            +factor(FACET.Joy>0)
            +factor(FACET.Confusion.to.Frustration>0),data=td)
anova(mcc_lm)
summary(mcc_lm)

mc_correct = td[,"MultipleChoiceScores"] == 1
a = td[mc_correct, "MultipleChoiceConfidence"]
b = td[mc_correct==FALSE,"MultipleChoiceConfidence"]
pooled_sd_num = (length(a)-1) * sd(a)^2 + (length(b) - 1) * sd(b)^2
pooled_sd_den = length(a) + length(b) - 2
pooled_sd = sqrt(pooled_sd_num / pooled_sd_den)
d = (mean(a) - mean(b))/pooled_sd; d
t.test(a, b)

mc_conf = td[,"MultipleChoiceConfidence"] > median(td[,"MultipleChoiceConfidence"])
t.test(td[mc_conf,"MultipleChoiceScores"], td[mc_conf==FALSE,"MultipleChoiceScores"])

# Checking rows in which student correctly identified an irrelevant text or diagram for differences in affect proportion
cj_rows = td[,"TextJudgmentScore"]==1 & td[,"TextRelevancy"]==0 | td[,"DiagramJudgmentScore"]==1 & td[,"DiagramRelevancy"]==0
ij_rows = td[,"TextJudgmentScore"]!=1 & td[,"TextRelevancy"]==0 | td[,"DiagramJudgmentScore"]!=1 & td[,"DiagramRelevancy"]==0
t.test(td[cj_rows, "FACET.Confusion"], td[ij_rows, "FACET.Confusion"])
t.test(td[cj_rows, "FACET.Confusion.Freq"], td[ij_rows, "FACET.Confusion.Freq"])
t.test(td[cj_rows, "FACET.Frustration"], td[ij_rows, "FACET.Frustration"])
t.test(td[cj_rows, "FACET.Frustration.Freq"], td[ij_rows, "FACET.Frustration.Freq"])
t.test(td[cj_rows, "L.FACET.Confusion.to.Frustration"], td[ij_rows, "L.FACET.Confusion.to.Frustration"])
t.test(td[cj_rows, "MultipleChoiceScores"], td[ij_rows, "MultipleChoiceScores"])

# Checking rows in which an agent shows congruence or incongruence for differences in affect proportion
ag_rows = td[,"AgentCongruence"] == -1 
agc_rows = td[,"AgentCongruence"] == 1
t.test(td[ag_rows, "FACET.Confusion"], td[agc_rows, "FACET.Confusion"])
t.test(td[ag_rows, "FACET.Frustration"], td[agc_rows, "FACET.Frustration"])
t.test(td[ag_rows, "L.FACET.Confusion.to.Frustration"], td[agc_rows, "L.FACET.Confusion.to.Frustration"])
table(td[ag_rows, "MultipleChoiceScores"], td[agc_rows, "MultipleChoiceScores"])
td[td[,"MultipleChoiceScores"]==0.5,"MultipleChoiceScores"] = 0
table(td[,"AgentCongruence"], td[,"MultipleChoiceScores"])[1:3,2:3]
summary(table(td[,"AgentCongruence"],td[,"MultipleChoiceScores"])[1:3,2:3])

c_table = table(factor(td[,"FACET.Confusion"] != 0), td[,"MultipleChoiceScores"])[1:2,2:3]
apply(c_table, 2, sum)/sum(c_table)
c_table[1,]/apply(c_table, 1, sum)[1]
c_table[2,]/apply(c_table, 1, sum)[2]
summary(c_table)

cf_table = table(factor(td[,"L.FACET.Confusion.to.Frustration"] > 0), factor(td[,"MultipleChoiceScores"]))[1:2,2:3]
apply(cf_table, 2, sum)/sum(cf_table)
cf_table[1,]/apply(cf_table, 1, sum)[1]
cf_table[2,]/apply(cf_table, 1, sum)[2]
summary(cf_table)

jf_table = table(factor(td[,"L.FACET.Confusion.to.Joy"] > 0), factor(td[,"MultipleChoiceScores"]))
apply(jf_table, 2, sum)/sum(jf_table)
jf_table[1,]/apply(jf_table, 1, sum)[1]
jf_table[2,]/apply(jf_table, 1, sum)[2]
summary(jf_table)

f_table = table(factor(td[,"FACET.Frustration"] > 0), factor(td[,"MultipleChoiceScores"]))[1:2,2:3]
apply(f_table, 2, sum)/sum(f_table)
f_table[1,]/apply(f_table, 1, sum)[1]
f_table[2,]/apply(f_table, 1, sum)[2]
summary(f_table)

j_table = table(factor(td[,"FACET.Joy"] > 0), factor(td[,"MultipleChoiceScores"]))
apply(j_table, 2, sum)/sum(j_table)
j_table[1,]/apply(j_table, 1, sum)[1]
j_table[2,]/apply(j_table, 1, sum)[2]
summary(j_table)

caf_table = table(factor(td[,"FACET.Confusion"] > 0), factor(td[,"FACET.Frustration"] > 0), td[,"MultipleChoiceScores"])
caf_table[,,1]/sum(caf_table[,,1])
caf_table[,,2]/sum(caf_table[,,2])

table(factor(td[,"FACET.Confusion"] > 0), factor(td[,"FACET.Frustration"] > 0))


