import numpy as np
import scipy.stats as ss
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
from AgencyJournal.minor_functions import condition_map


def add_holm_bonferroni(p_value_list, index_match, sig_level=0.05):
    total_hypotheses = len(p_value_list)
    first_fail_to_reject = total_hypotheses  # This will reject all hypotheses, returing True for all p-values
    for k in range(1, total_hypotheses+1):
        i = k - 1
        revised_sig_level = sig_level / (total_hypotheses + 1 - k)
        if p_value_list[i] > revised_sig_level and first_fail_to_reject == total_hypotheses:
            first_fail_to_reject = i
    return_list = [True] * first_fail_to_reject + [False] * (total_hypotheses - first_fail_to_reject)
    return_series = pd.Series(return_list, index=index_match)
    return return_series


def posthoc(groups, response, show='minimum'):
    anova_results = anova(groups, response, show='maximum')
    print("---------------------------------------------")
    group_list = list(groups.unique())
    for group_index_a in range(len(group_list)-1):
        group_a_rows = groups == group_list[group_index_a]
        samp_a = response.loc[group_a_rows]
        for group_index_b in range(group_index_a + 1, len(group_list)):
            group_b_rows = groups == group_list[group_index_b]
            samp_b = response.loc[group_b_rows]
            print("Group:%s minus Group:%s" % (condition_map(group_list[group_index_a]), condition_map(group_list[group_index_b])))
            welchs_ttest(samp_a, samp_b, show='maximum')
            print("-----------------------------------------")


def welchs_ttest(samp_one, samp_two, show="minimum"):
    t_num = samp_one.mean() - samp_two.mean()
    n_one = len(samp_one)
    n_two = len(samp_two)
    pooled_sd = np.sqrt((n_one * samp_one.std() ** 2 + n_two * samp_two.std() ** 2) / (n_one + n_two))
    denom_var = samp_one.var() / n_one + samp_two.var() / n_two
    t_stat = t_num / np.sqrt(denom_var)
    df = denom_var ** 2 / (samp_one.var()**2 / (n_one**2 * (n_one-1)) + samp_two.var()**2 / (n_two**2 * (n_two-1)))
    df = math.floor(df)
    p_low = ss.t.cdf(t_stat, df=df)
    p_val = min(p_low, 1 - p_low) * 2
    effect_size = t_num / pooled_sd
    if show == "maximum" or show == "medium":
        print("t-stat(%d): %.4f" % (df, t_stat))
        print("p-val : %.4f" % p_val)
        print("effect: %.4f" % effect_size)
    return pd.Series(data=[t_stat, df, p_val, effect_size], index=["t-stat", "df", "p-value", "Effect"])


def ttest(samp_one, samp_two, show="minimum"):
    t_num = samp_one.mean() - samp_two.mean()
    n_one = len(samp_one) - 1
    n_two = len(samp_two) - 1
    df = n_one + n_two
    pooled_sd = np.sqrt((n_one * samp_one.std()**2 + n_two * samp_two.std()**2) / (n_one + n_two))
    t_den = pooled_sd * np.sqrt(1.0 / (n_one + 1) + 1.0 / (n_two + 1))
    t_stat = t_num / t_den
    effect_size = t_num / pooled_sd
    p_val = 1 - ss.t.cdf(t_stat, df=df)
    if show == "maximum" or show == "medium":
        print("t-stat(%d): %.4f" % (df, t_stat))
        print("p-val : %.4f" % p_val)
        print("effect: %.4f" % effect_size)
    return pd.Series(data=[t_stat, df, p_val, effect_size], index=["t-stat", "df", "p-value", "Effect"])


def anova(groups, response, show="minimum"):
    """Manually calculating an ANOVA under an arbitrary number of groups
    :param groups: Pandas Series with discrete labels for each group
    :param response: Pandas Series corresponding to groups with the response variable for each individual
    :returns """
    all_groups = list(groups.unique())
    N = len(groups)
    G = len(all_groups)
    overall_mean = response.mean()

    ss_treatment = 0
    ss_error = 0
    for g in all_groups:
        response_group = response.loc[groups == g]
        group_mean = response_group.mean()
        group_n = response_group.shape[0]
        ss_treatment += group_n * (group_mean - overall_mean)**2
        ss_error += np.sum((response_group - group_mean)**2)
        if show == "maximum":
            print("Group %d (n=%d) Mean:%.4f SD: %.4f" % (g, group_n, group_mean, response_group.std()))

    df_one = (G - 1)
    df_two = (N - G)
    ms_treatment = ss_treatment / df_one
    ms_error = ss_error / df_two
    F_stat = ms_treatment / ms_error
    p_val = 1 - ss.f.cdf(F_stat, df_one, df_two)

    ss_total = np.sum((response - overall_mean)**2)  # Should be equal to ss_treatment + ss_error

    if show == "maximum" or "medium":
        print("SSTreat:%.4f" % ss_treatment)
        print("SSError:%.4f" % ss_error)
        print("F-stat(%d, %d):%.4f" % (df_one, df_two, F_stat))
        print("p-val:%.4f" % p_val)
        print("SSTotal:%.4f" % ss_total)

    return pd.Series(data=[ss_treatment, ss_error, F_stat, ss_total, p_val],
                     index=["SSTreat", "SSError", "F-stat", "SSTotal", "p-val"])


def ancova(groups, response, covariate, show='minimum'):
    # Currently only works for one covariate
    df = pd.DataFrame()
    df["Group"] = groups
    df["Response"] = response
    df["Covariate"] = covariate
    ancova_results = smf.ols("Response ~ Covariate + C(Group)", data=df).fit()
    f_stat = ancova_results.fvalue
    df_model = ancova_results.df_model
    df_resid = ancova_results.df_resid
    p_val = ancova_results.f_pvalue
    if show == 'maximum':
        print(ancova_results.summary())
    elif show == 'medium':
        print("F-stat (%.3f, %.3f): %.4f" % (df_model, df_resid, f_stat))
        print("p-value: %.4f" % p_val)

    return pd.Series(data=[f_stat, df_model, df_resid, p_val],
                     index=["F-stat", "df1", "df2", "p-value"])


def simulation_test():
    students_per_cond = 10000
    # time variables dependent on agency condition
    times_full = np.random.normal(loc=4168.6/60.0, scale=1317/60.0, size=students_per_cond)
    times_part = np.random.normal(loc=4987.8/60.0, scale=1094/60.0, size=students_per_cond)
    times_none = np.random.normal(loc=5465/60.0, scale=0, size=students_per_cond)
    nlg_samps = np.random.normal(loc=0.256, scale=0.2636, size=students_per_cond*3)

    # independent variables from NLG distribution
    nlg_full = nlg_samps[:students_per_cond]
    nlg_part = nlg_samps[students_per_cond:students_per_cond*2]
    nlg_none = nlg_samps[students_per_cond*2:students_per_cond*3]

    # transformed variables from time
    nlg_time_full = nlg_full / times_full
    nlg_time_part = nlg_part / times_part
    nlg_time_none = nlg_none / times_none

    condition = np.array([1] * students_per_cond + [2] * students_per_cond + [3] * students_per_cond)
    df = pd.DataFrame(columns=["Condition", "Time", "NLG"])
    df["Condition"] = condition
    df["Time"] = np.concatenate([times_full, times_part, times_none])
    df["NLG"] = np.concatenate([nlg_full, nlg_part, nlg_none])
    df["NLGstd"] = np.concatenate([nlg_time_full, nlg_time_part, nlg_time_none])
    ancova_nlg_results = smf.ols("NLG ~ Time + C(Condition)", data=df).fit()
    print(ancova_nlg_results.summary())
    ancova_nlgt_results = smf.ols("NLGstd ~ Time + C(Condition)", data=df).fit()
    print(ancova_nlgt_results.summary())

    anova_nlg_results = smf.ols("NLG ~ C(Condition)", data=df).fit()
    print(anova_nlg_results.summary())
    anova_nlgt_results = smf.ols("NLGstd ~ C(Condition)", data=df).fit()
    print(anova_nlgt_results.summary())

    anova_time_results = smf.ols("Time ~ C(Condition)", data=df).fit()
    print(anova_time_results.summary())

