mvn_KL <- function(mu0, mu1, v0, v1, dim){
	inv1 = solve(v1)
	t1 = sum(diag(inv1 %*% v0))
	t2 = t(mu1 - mu0) %*% inv1 %*% (mu1 - mu0)
	t3 = -dim + log(det(v1)/det(v0))
	kld = 0.5 * (t1 + t2 + t3)
	return(kld[1,1])
}

sym_mvn_KL <- function(mu0, mu1, v0, v1, dim){
	d1 = mvn_KL(mu0, mu1, v0, v1, dim)
	d2 = mvn_KL(mu1, mu0, v1, v0, dim)
	symd = 0.5 * d1 + 0.5 * d2
	return(list(d1=d1,d2=d2,symmetric=symd))
}

dist_test <- function(mu0, mu1, mua, v0, v1, va, dim, n){
	numerator = sym_mvn_KL(mu0, mu1, v0, v1, dim)$symmetric
	denom1 = sym_mvn_KL(mu0, mua, v0, va, dim)$symmetric
	denom2 = sym_mvn_KL(mu1, mua, v1, va, dim)$symmetric
	denominator = 0.5 * (denom1 + denom2)
	F_stat = numerator / denominator
	df1 = dim
	df2 = n - dim
	p = 1 - pf(F_stat, df1, df2)
	return(list(F=F_stat, df1=df1, df2=df2, p=p))
}

under_thresh <- function(diff_vec, threshold){
	outside = sum(abs(diff_vec)>threshold)
	return(outside > 0)
}

# This is a Bayesian test of whether at least one mean in two MultiVar Normal differ by threshold
sample_test <- function(mu0, mu1, v0, v1, threshold){
	n = 10000
	s1 = mvrnorm(n,mu0,v0)
	s2 = mvrnorm(n,mu1,v1)
	outside = apply(s1-s2, 1, FUN=under_thresh, threshold=threshold)
	out.prop = sum(outside)/n
	return(out.prop)
}