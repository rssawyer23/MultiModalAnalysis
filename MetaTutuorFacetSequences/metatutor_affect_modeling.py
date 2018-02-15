# Script for AIED2018 (and beyond) modeling using the affect events as features for different response variables

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

data_filepath = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Probabilities-Trial.csv"
data = pd.read_csv(data_filepath)
valid_rows = data.loc[:, "MultipleChoiceScores"] != -1
data = data.loc[valid_rows, :]
valid_rows = data.loc[:,"TestSubject"] != "IVH2_PN050"
data = data.loc[valid_rows, :]
data.index=list(range(data.shape[0]))
valid_rows = data.loc[:,"MultipleChoiceScores"] == 1
correct_indices = np.where(valid_rows)[0]
np.random.shuffle(correct_indices)
remove_correct = correct_indices[402:]
data = data.drop(labels=remove_correct, axis=0)
data = data.replace(np.nan, 0)
data.index = list(range(data.shape[0]))

samp_weights = np.ones(data.shape[0])
samp_weights[data["MultipleChoiceScores"] == 0] = 2.307
samp_weights[data["MultipleChoiceScores"] == 0.5] = 1.154

data["SampWeights"] = samp_weights

partial_correct = data.loc[:, "MultipleChoiceScores"] == 0.5
data.loc[partial_correct, "MultipleChoiceScores"] = 0
data.loc[:,"MultipleChoiceScores"] = data.loc[:,"MultipleChoiceScores"].astype('int')



for emotion in ["Confusion", "Frustration", "Joy"]:
    data["%s-Present" % emotion] = pd.Series(data["FACET-%s" % emotion] > 0, dtype='int32')
data["CtoF-Present"] = pd.Series(data["FACET-Confusion-to-Frustration"] > 0, dtype='int32')

logreg = LogisticRegression()
dtree = DecisionTreeClassifier(criterion='gini', max_depth=2)
print(data.shape)
kf = KFold(n_splits=50, shuffle=True)
kf.get_n_splits(data)

# PREDICTED IS BY COLUMN, ACTUAL IS BY ROW
# aka cell (0,2) is times predicted class index 2 when actual was class index 0
logreg_cm = np.zeros((2, 2))
dtree_cm = np.zeros((2, 2))

features = ["FACET-Confusion-Freq", "FACET-Frustration-Freq", "FACET-Joy-Freq", "CtoF-Present"]

for train_index, test_index in kf.split(data):
    X_train = data.loc[train_index, features]
    Y_train = data.loc[train_index, "MultipleChoiceScores"]

    logreg.fit(X=X_train, y=Y_train)
    dtree.fit(X=X_train, y=Y_train)

    X_test = data.loc[test_index, features]
    Y_test = data.loc[test_index, "MultipleChoiceScores"]

    logreg_preds = logreg.predict(X_test)
    logreg_cm_fold = confusion_matrix(Y_test, logreg_preds)
    logreg_cm += logreg_cm_fold
    dtree_preds = dtree.predict(X_test)
    dtree_cm_fold = confusion_matrix(Y_test, dtree_preds)
    dtree_cm += dtree_cm_fold

print("PREDICTED IS BY COLUMN, ACTUAL IS BY ROW")
print("Baseline: %.4f" % np.mean(data.loc[:,"MultipleChoiceScores"]))
print("Logistic Regression Confusion Matrix:")
print(logreg_cm)
print("Accuracy: %.4f" % (np.trace(logreg_cm) / np.sum(logreg_cm)))
logreg.fit(X=data.loc[:, features], y=data.loc[:,"MultipleChoiceScores"], sample_weight=np.array(data.loc[:,"SampWeights"]))
print(pd.Series(logreg.coef_[0], index=features))

print("Decision Tree Confusion Matrix:")
print(dtree_cm)
print("Accuracy: %.4f" % (np.trace(dtree_cm) / np.sum(dtree_cm)))

