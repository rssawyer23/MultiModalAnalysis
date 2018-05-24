import pymc3 as pm
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reflect_add_cols = ['C-DP-Prompts', 'C-DP-Posters', 'MeanHowIsItGoingLikert']
feature_cols_props = ['FinalGameScore', 'MysterySolved', 'C-DP-Conversation', 'C-DP-BooksAndArticles',
                      'C-DP-Worksheet', 'C-DP-Scanner', 'C-A-PlotPoint', 'C-A-WorksheetSubmit']
features = feature_cols_props + reflect_add_cols


def create_ind_hier_movement():
    pass


def create_comparison_plot(h_samps, feature, feature_list, save_filebase):
    # Get the samples for the feature for each group
    feature_index = feature_list.index(feature)
    reflect_samples = h_samps['beta_reflect'][:, feature_index]
    leads_samples = h_samps['beta_leads'][:, feature_index]

    # Calculate probability Classroom > Laboratory
    samples = len(reflect_samples)
    p = (leads_samples > reflect_samples).sum() / samples

    # Plot samples and text for p-value comparison
    fig, ax = plt.subplots(1)
    sns.distplot(leads_samples, hist=False, kde=True, rug=False, color='b', label='Laboratory', ax=ax)
    sns.distplot(reflect_samples, hist=False, kde=True, rug=False, color='r', label='Classroom', ax=ax)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Posterior Density")
    ax.text(x=0.7, y=1.2, s="Pr(Lab > Class) = %.3f" % p)
    ax.legend()
    plt.savefig(save_filebase + feature + 'Comparison.pdf')
    plt.close()


sample_directory = "C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/REFLECT/saved_samples/"
with open(sample_directory+"pooled_5000.pkl", 'rb') as buff:
    pool_samps = pickle.load(buff)
with open(sample_directory+"individual_5000.pkl", 'rb') as buff:
    ind_samps = pickle.load(buff)
with open(sample_directory+"hierarchical_5000.pkl", 'rb') as buff:
    hier_samps = pickle.load(buff)

index_list = ["Intercept"] + features + ["Uncertainty"]
ps_df = pm.df_summary(pool_samps)
ps_df.index = index_list
hs_df = pm.df_summary(hier_samps)
in_df = pm.df_summary(ind_samps)

reflect_indices = [e for e in hs_df.index if 'reflect' in e]
rh_df = hs_df.loc[reflect_indices, :]
rh_df.index = ["%s" % e for e in index_list]
ri_df = in_df.loc[reflect_indices, :]
ri_df.index = ["%s" % e for e in index_list]

leads_indices = [e for e in hs_df.index if 'leads' in e]
lh_df = hs_df.loc[leads_indices, :]
lh_df.index = ["%s" % e for e in index_list]
li_df = in_df.loc[leads_indices, :]
li_df.index = ["%s" % e for e in index_list]

p = (hier_samps['alpha_reflect'] > hier_samps['alpha_leads']).sum()/5000
significant = p < 0.025 or p > 0.975
print("\tp-value=%.4f Significant:%s" % (p, significant))
p = (hier_samps['sigma_y_reflect'] > hier_samps['sigma_y_leads']).sum()/5000
significant = p < 0.025 or p > 0.975
print("\tp-value=%.4f Significant:%s" % (p, significant))
for i in range(len(features)):
    print("Comparison with %s" % features[i])
    p = (hier_samps['beta_reflect'][:, i] > hier_samps['beta_leads'][:, i]).sum()/5000
    significant = p < 0.025 or p > 0.975
    print("\tp-value=%.4f Significant:%s" % (p, significant))

for i in range(len(features)):
    print("Comparison with %s" % features[i])
    p = (ind_samps['beta_reflect'][:, i] > ind_samps['beta_leads'][:, i]).sum()/5000
    significant = p < 0.025 or p > 0.975
    print("\tp-value=%.4f Significant:%s" % (p, significant))
pm.df_summary(ind_samps)

create_comparison_plot(hier_samps,
                       feature="FinalGameScore",
                       feature_list=features,
                       save_filebase="C:/Users/robsc/Documents/NC State/GRAWork/Publications/AIIDE18-Transfer/Images/")
