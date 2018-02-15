# Analyzing the Crystal Island Affect data for UMUAI Special Issue on Affect due 3/15/2018
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix


def calculate_vif(data, cols, col):
    lm = LinearRegression()
    t_list = list(cols)
    t_list.remove(col)
    lm.fit(data[t_list],data[col])
    r2 = lm.score(data[t_list],data[col])
    vif = 1. / (1 - r2)
    return vif


def overall_summary(data, show=True):
    if show:
        print("Number of students: %d" % data.shape[0])


def response_summaries(data, response_list, show=True):
    for response in response_list:
        if show:
            print(data[response].describe())
            print("Positive(%%): %d (%.3f%%)" % (np.sum(data[response] > 0), np.mean(data[response] > 0)*100))


def independent_summaries(data, cols, show=True):
    if show:
        print(data[cols].describe())
        print(data[cols].corr(method='pearson'))
    for col in cols:
        col_vif = calculate_vif(data, cols, col)
        print("%s VIF: %.4f" % (col, col_vif))


def composite_tests(data, response_name, ind_names, show=True):
    response_col = "%sEvidence-0-Sum" % response_name
    ind_cols = ["%sEvidence-0-Sum" % e for e in ind_names]
    lm = LinearRegression()
    lm.fit(data[ind_cols], data[response_col])
    r2 = lm.score(data[ind_cols], data[response_col])
    if show:
        print("%s from AUs %s R2: %.3f" % (response_name, ind_names, r2))
        for c, n in zip(lm.coef_, ind_names):
            print("%s: %.3f" % (n, c))


def all_summary_statistics(data, show=True):
    AU_cols = [e for e in data.columns.values if "AU" in e]
    Comp_cols = [e for e in data.columns.values if "AU" not in e and "Evidence" in e]
    Game_cols = [e for e in data.columns.values if "Duration" in e or "C-" in e or "CD-" in e]

    print(
        "----------------------------------------------\n|Action Unit Summaries\n-------------------------------------------")
    # Outputting analysis for paper writing
    overall_summary(data, show)
    response_summaries(data, response_list=["NormalizedLearningGain", "Presence"], show=show)
    independent_summaries(data, cols=AU_cols, show=show)
    print(
        "----------------------------------------------\n|Composite Summaries\n-------------------------------------------")
    independent_summaries(data, cols=Comp_cols, show=show)
    print(
        "----------------------------------------------\n|Game Feature Summaries\n-------------------------------------------")
    independent_summaries(data, cols=Game_cols, show=show)

    print(
        "----------------------------------------------\n|Composites from AUs\n-------------------------------------------")
    # Outputting breakdown of composites by contributing AU linear model
    composite_tests(data, response_name="Joy", ind_names=["AU6", "AU12"], show=show)
    composite_tests(data, response_name="Sadness", ind_names=["AU1", "AU4", "AU15"], show=show)
    composite_tests(data, response_name="Surprise", ind_names=["AU1", "AU2", "AU5", "AU26"], show=show)
    composite_tests(data, response_name="Fear", ind_names=["AU2", "AU4", "AU5", "AU7", "AU20", "AU26"], show=show)
    composite_tests(data, response_name="Anger", ind_names=["AU4", "AU5", "AU7", "AU23"], show=show)
    composite_tests(data, response_name="Disgust", ind_names=["AU9", "AU15"], show=show)
    composite_tests(data, response_name="Contempt", ind_names=["AU12", "AU14"], show=show)


def add_interaction_terms(data, col_set1, col_set2):
    new_data = data.copy()
    col_list = []
    for col in col_set1:
        for col2 in col_set2:
            new_data["%s-%s" % (col[:5], col2[:5])] = data[col] * data[col2]
            new_data["%s-%s" % (col[:5], col2[:5])] = (new_data["%s-%s" % (col[:5], col2[:5])] - new_data["%s-%s" % (col[:5],col2[:5])].mean())/(new_data["%s-%s" % (col[:5],col2[:5])].std())
            col_list.append("%s-%s" % (col[:5], col2[:5]))
    return new_data, col_list


def cv_regression(model, data, feature_cols, response_col, show=False):
    kf = KFold(n_splits=data.shape[0])
    rss = 0
    for train, test, in kf.split(data):
        model.fit(X=data.loc[train, feature_cols], y=data.loc[train, response_col])
        pred = model.predict(X=data.loc[test, feature_cols])[0]
        rss += (pred - data.loc[test, response_col].iloc[0]) ** 2
    tss = np.sum((data[response_col] - data[response_col].mean())**2)
    cvR2 = 1 - rss / tss
    model.fit(data.loc[:,feature_cols], data.loc[:, response_col])
    r2 = model.score(X=data.loc[:, feature_cols], y=data.loc[:, response_col])
    if show:
        print("CVR2: %.4f" % np.mean(cvR2))
        print("R2: %.4f" % r2)
        # for c, n in zip(model.coef_, feature_cols):
        #     if abs(c) > 0.0001:
        #         print("%s: %.5f" % (n, c))
    return cvR2, r2


def cv_classification(model, data, feature_cols, response_col, show=False):
    kf = KFold(n_splits=data.shape[0])
    total_correct = 0
    y_pred = []
    y_true = []
    for train, test in kf.split(data):
        model.fit(X=data.loc[train, feature_cols], y=data.loc[train, response_col])
        pred = model.predict(X=data.loc[test, feature_cols])[0]
        total_correct += int(np.equal(pred, data.loc[test, response_col].iloc[0]))
        y_pred.append(pred)
        y_true.append(data.loc[test, response_col].iloc[0])
    if show:
        print(confusion_matrix(y_pred=y_pred, y_true=y_true))
    test_accuracy = float(total_correct) / data.shape[0]
    train_accuracy = model.score(X=data.loc[:,feature_cols], y=data.loc[:,response_col])
    return test_accuracy, train_accuracy


def ens_classification(model, data, feature_sets, response_col, show=False):
    kf = KFold(n_splits=data.shape[0])
    total_correct = 0
    y_pred = []
    y_true = []
    for train, test in kf.split(data):
        pred_probs = np.array([0] * len(data[response_col].unique()),dtype='float64')
        for f_set in feature_sets:
            model.fit(X=data.loc[train, f_set], y=data.loc[train, response_col])
            pred_probs += model.predict_proba(X=data.loc[test, f_set])[0]
        pred = np.argmax(pred_probs)
        total_correct += int(np.equal(pred, data.loc[test, response_col].iloc[0]))
        y_pred.append(pred)
        y_true.append(data.loc[test, response_col].iloc[0])
    test_accuracy = float(total_correct) / data.shape[0]
    return test_accuracy, -1.0


def multivariate_regression(output_filename):
    regression_output = open(output_filename, 'w')

    lm = MultiTaskLasso(alpha=0.1)
    reg_name = "MTLassoRegression"

    gcvr2, gr2 = cv_regression(lm, n_data, Game_cols, ["NormalizedLearningGain", "Presence"], show=True)
    gccvr2, gcr2 = cv_regression(lm, n_data, Game_cols + Comp_cols, ["NormalizedLearningGain", "Presence"], show=True)
    gaucvr2, gaur2 = cv_regression(lm, n_data, Game_cols + AU_cols, ["NormalizedLearningGain", "Presence"], show=True)


def generate_binary_cols(df, cols):
    new_cols = []
    for col in cols:
        df["%s-Binary" % col] = np.array(df[col] > df[col].median(), dtype=int)
        new_cols.append("%s-Binary" % col)
    return df, new_cols


def classification(output_filename):
    classification_output = open(output_filename, 'w')
    classification_output.write("Classifier,ClassBreakdown,GameTest,GameTrain,CompTest,CompTrain,AUTest,AUTrain,BCompTest,BCompTrain,BAUTest,BAUTrain\n")

    classifiers = [LogisticRegression(C=0.1),
                   LogisticRegression(C=3.0),
                   SVC(C=0.1, kernel='linear'),
                   GaussianNB()
                   ]
    classifier_names = ["LogReg.1", "LogReg3", "SVC", "NB"]

    for cl, c_name in zip(classifiers, classifier_names):
        for class_breakdown in ["MedianNLG", "MedianPres", "FourClass", "ThreeClass"]:
            feature_accuracies = []
            for f_set in [Game_cols, Game_cols + Comp_cols, Game_cols + AU_cols, Game_cols + Comp_cols + AU_cols]:
                test_acc, train_acc = cv_classification(cl, n_data, f_set, class_breakdown, show=show)
                feature_accuracies.append("%.3f" % test_acc)
                feature_accuracies.append("%.3f" % train_acc)
            classification_output.write("%s,%s,%s\n" % (c_name, class_breakdown,",".join(feature_accuracies)))

    # for cl, c_name in zip(classifiers, classifier_names):
    #     for class_breakdown in ["MedianNLG", "MedianPres", "FourClass", "ThreeClass"]:
    #         feature_accuracies = []
    #         f_set = [Game_cols + Comp_cols, Game_cols + AU_cols, Game_cols + bin_Comp, Game_cols + bin_AU]
    #         test_acc, train_acc = ens_classification(cl, n_data, f_set, class_breakdown, show=show)
    #         feature_accuracies.append("%.3f" % test_acc)
    #         feature_accuracies.append("%.3f" % train_acc)
    #         classification_output.write("%s,%s,%s\n" % (c_name, class_breakdown,",".join(feature_accuracies)))


def univariate_regression(output_filename):
    regression_output = open(output_filename, 'w')
    regression_output.write(
        "ModelName,Response,GameplayCVR2,CompositeCVR2,Composite-InteractionCVR2,AUCVR2,AU-InteractionCVR2\n")
    regressors = [LinearRegression(), Lasso(alpha=0.1), Ridge(alpha=1.0),
                  MLPRegressor(hidden_layer_sizes=(20, 5), alpha=0.0001, early_stopping=True),
                  RandomForestRegressor(), KNeighborsRegressor(n_neighbors=7, weights='distance')]
    regressor_names = ["LinearRegression", "Lasso", "Ridge", "MLP", "RandomForestReg", "KNN"]

    for lm, r_name in zip(regressors, regressor_names):
        show = r_name != "MLP" and r_name != "RandomForestReg" and r_name != "KNN"

        print(
            "------------------------------------\nPredicting NLG from Game Cols using %s\n----------------------------------------" % r_name)
        gr2 = cv_regression(lm, n_data, Game_cols, "NormalizedLearningGain", show=show)
        print(
            "------------------------------------\nPredicting NLG from Game Cols + Comp Cols using %s\n------------------------------" % r_name)
        gc_data, gc_cols = add_interaction_terms(n_data, Game_cols, Comp_cols)
        cir2 = cv_regression(lm, gc_data, gc_cols, "NormalizedLearningGain", show=show)
        cr2 = cv_regression(lm, n_data, Game_cols + Comp_cols, "NormalizedLearningGain", show=show)
        print(
            "------------------------------------\nPredicting NLG from Game Cols + AU Cols using %s\n----------------------------------------" % r_name)
        gau_data, gau_cols = add_interaction_terms(n_data, Game_cols, AU_cols)
        auir2 = cv_regression(lm, gau_data, gau_cols, "NormalizedLearningGain", show=show)
        aur2 = cv_regression(lm, n_data, Game_cols + AU_cols, "NormalizedLearningGain", show=show)
        line = "%s,NLG,%.3f,%.3f,%.3f,%.3f,%.3f" % (r_name, gr2, cr2, cir2, aur2, auir2)
        regression_output.write(line + "\n")

        print(
            "------------------------------------\nPredicting Presence from Game Cols using %s\n----------------------------------------" % r_name)
        gr2 = cv_regression(lm, n_data, Game_cols, "Presence", show=show)
        print(
            "------------------------------------\nPredicting Presence from Game Cols + Comp Cols using %s\n------------------------------" % r_name)
        cir2 = cv_regression(lm, gc_data, gc_cols, "Presence", show=show)
        cr2 = cv_regression(lm, n_data, Game_cols + Comp_cols, "Presence", show=show)
        print(
            "------------------------------------\nPredicting Presence from Game Cols + AU Cols using %s\n----------------------------------------" % r_name)
        auir2 = cv_regression(lm, gau_data, gau_cols, "Presence", show=show)
        aur2 = cv_regression(lm, n_data, Game_cols + AU_cols, "Presence", show=show)
        line = "%s,Presence,%.3f,%.3f,%.3f,%.3f,%.3f" % (r_name, gr2, cr2, cir2, aur2, auir2)
        regression_output.write(line + "\n")

if __name__ == "__main__":
    # Script parameter settings
    show = True

    # Reading and cleaning data
    data_filename = "Data/Positive_Spike_SummaryPost_Appended_Std_Full.csv"
    data = pd.read_csv(data_filename)

    include_partial = False
    if not include_partial:
        cond = data["Condition"]
        keep_rows = data["Condition"] == "Full"
        data = data.loc[keep_rows,:]
        data.drop(["TestSubject", "Condition"], axis=1, inplace=True)

    n_data = data.apply(lambda col: (col - np.mean(col)) / np.std(col))
    AU_cols = [e for e in data.columns.values if "AU" in e]
    Comp_cols = [e for e in data.columns.values if "AU" not in e and "Evidence" in e]
    Game_cols = [e for e in data.columns.values if "Duration" in e or "C-" in e or "CD-" in e]
    n_data, bin_Comp = generate_binary_cols(n_data, Comp_cols)
    n_data, bin_AU = generate_binary_cols(n_data, AU_cols)

    if include_partial:
        n_data["nCondition"] = np.array(cond == "Full", dtype=int)
        Game_cols.append("nCondition")

    # Summary statistics
    #all_summary_statistics(data, show=show)

    # Regression Models
    #univariate_regression(output_filename="RegressionResults.csv")
    #multivariate_regression(output_filename="MVRegressionResults.csv")

    # Classification Models
    n_data["MedianNLG"] = np.array(n_data["NormalizedLearningGain"] > n_data["NormalizedLearningGain"].median(), dtype=int)
    n_data["MedianPres"] = np.array(n_data["Presence"] > n_data["Presence"].median(), dtype=int)
    n_data["FourClass"] = n_data["MedianNLG"]*2 + n_data["MedianPres"]
    n_data["ThreeClass"] = n_data["MedianNLG"] + n_data["MedianPres"]
    print(n_data["FourClass"].value_counts())
    print(max(n_data["FourClass"].value_counts())/n_data.shape[0])
    print(n_data["ThreeClass"].value_counts())
    print(max(n_data["ThreeClass"].value_counts())/n_data.shape[0])
    classification(output_filename="ClassificationResultsPost.csv")
