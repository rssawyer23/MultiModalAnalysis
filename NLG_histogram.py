import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

act_sum = pd.read_csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryAppendedEdited.csv")
nlg = act_sum.loc[:, "NLG"]

fig, ax = plt.subplots(1)
array = np.arange(-0.6, 0.7, 0.1)
array[8] = 0.181
ax.hist(nlg, bins=array)
ax.axvline(x=0.182, color='red')
ax.set_xlabel("Normalized Learning Gain")
ax.set_ylabel("Students")
ax.set_title("Normalized Learning Gain Histogram")
print(np.sum(nlg < 0.181))
print(np.sum(nlg > 0.181))
plt.show()
