library(MASS)
path = "C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/Data/Positive_Spike_SummaryPost_Appended_Std.csv"
data = read.csv(path)
data["nCondition"] = 0
data[data["Condition"] == "Full","nCondition"] = 1
game = c(names(data)[grepl("C.",names(data),fixed=TRUE) | grepl("CD.",names(data),fixed=TRUE)],"DurationInterval.012","nCondition")
comp = names(data)[grepl("Evidence",names(data),fixed=TRUE) & !grepl("AU",names(data),fixed=TRUE)]
aus = names(data)[grepl("AU",names(data),fixed=TRUE)]

game_data = scale(data[game])
gc_data = scale(data[c(game,comp)])
gau_data = scale(data[c(game,aus)])
nlg = scale(data["NormalizedLearningGain"])
pres = scale(data["Presence"])

loocv_r2 <- function(df, response){
	rss = 0
	for(i in 1:dim(df)[1]){
		tlm = lm(response[-i]~.,data=as.data.frame(df[-i,]))
		t_step = step(tlm, direction="both",k=log(dim(df[-i,])[1]))
		pred = predict(t_step,as.data.frame(t(df[i,])))
		squared_error = (response[i] - pred)^2
		rss = rss + squared_error
	}
	tlm = lm(response~.,data=as.data.frame(df))
	t_step = step(tlm, direction="both",k=log(dim(df)[1]))
	r2 = summary(t_step)$r.squared	
	tss = sum((response - mean(response))^2)
	ar2 = 1 - sum((response - predict(t_step))^2) / tss
	cvr2 = 1 - rss/tss
	to_return = list(CV_R2=cvr2, R2=r2, AR2=ar2)
	to_return
}
game_cvr2_nlg = loocv_r2(game_data, nlg)
game_cvr2_pre = loocv_r2(game_data, pres)
gc_cvr2_nlg = loocv_r2(gc_data, nlg)
gc_cvr2_pre = loocv_r2(gc_data, pres)
gau_cvr2_nlg = loocv_r2(gau_data,nlg)
gau_cvr2_pre = loocv_r2(gau_data,pres)

game_cvr2_nlg
game_cvr2_pre
gc_cvr2_nlg
gc_cvr2_pre
gau_cvr2_nlg
gau_cvr2_pre
