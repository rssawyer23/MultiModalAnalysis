# Transfer models using shared data between REFLECT Feb18 study and LEADS Full study
# First target venue: AIIDE-2018 (special theme: across physical spaces)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge # For cross validation
import statsmodels.api as sm  # For descriptive model
from sklearn.model_selection import KFold
import REFLECT.transfer_pymc as tpmc
import pickle


def print_demographics(df, response, title_key):
    total_students = df.shape[0]
    females = sum(df["Gender"] == 2) if str(df.loc[0, "Gender"]).isdigit() else sum(df["Gender"] == "Female")
    finished = sum(df["MysterySolved"])
    pos_lg = sum(df["RevisedLG"] > 0) if "RevisedLG" in df.columns else sum(df["LearningGain"] > 0)
    zero_lg = sum(df["RevisedLG"] == 0) if "RevisedLG" in df.columns else sum(df["LearningGain"] == 0)
    warning = "WARNING: " if title_key == "BOTH" else ""
    print("-----------------------------------------------------------------------------------------------")
    print("%s Demographic Data" % title_key)
    print("\t%d Students Total" % total_students)
    print("\t%s%d Females (%.4f)" % (warning, females, females/total_students))
    print("\t%d solved the mystery (%.4f)" % (finished, finished/total_students))
    print("\t%s%d had positive learning gains (%.4f), %d had no learning gains (%.4f)" %
          (warning, pos_lg, pos_lg/total_students, zero_lg, zero_lg/total_students))
    print(df.loc[:, [response, "Age", "Duration"]].describe().loc[["mean", "std", "min", "max"]].T)
    print("-----------------------------------------------------------------------------------------------")


def print_descriptive_lm(df, feature_cols, response_col):
    Xdf = df.loc[:, feature_cols]
    X = sm.add_constant(Xdf)
    y = df.loc[:, response_col]
    reflect_ols = sm.OLS(y, X).fit()
    print(reflect_ols.summary())


def preprocess_df(df, feature_cols, scale=False):
    """Convert duration to minutes and divide counts by duration to get rates (and proportions)"""
    scaler = StandardScaler()

    df["Duration"] = df["Duration"] / 60
    df["MysterySolved"] = pd.Series(df["MysterySolved"], dtype=int)

    df.fillna(df.mean(), inplace=True)

    for col in [e for e in df.columns if "C-D-" in e]:
        df[col.replace("C-D-", "C-DP-")] = df[col] / df["Duration"]  # Calculating Duration Proportions (DP)

    if scale:  # Should generally be doing this during training folds only (hence default:False)
        df.loc[:, feature_cols] = scaler.fit_transform(df.loc[:, feature_cols])

    return df, scaler


def evaluate_cv_performance(X, y, model, folds, show=True):
    """Function for evaluating (model) on independent variables (X) and response (y) 
    using cross-validation with (folds) number of folds"""
    kf = KFold(n_splits=folds)
    errors = []
    cvm_errors = []
    for train_indices, test_indices in kf.split(X):
        # Parse the fold data into train/test splits and scale using only the training data (but transform the test data)
        X_scaler, y_scaler = StandardScaler(), StandardScaler()
        X_train, y_train = X_scaler.fit_transform(X.loc[train_indices, :]), y_scaler.fit_transform(y.loc[train_indices].values.reshape(-1,1))
        X_test, y_test = X_scaler.transform(X.loc[test_indices, :].values.reshape(1,-1)), y_scaler.transform(y.loc[test_indices].values.reshape(-1,1))

        # Train model on training data
        model.fit(X=X_train, y=y_train)

        # Predict the held out test data
        predictions = model.predict(X=X_test)

        # Append the errors from the model's predictions on the test set
        fold_errors = y_test - predictions
        errors += list(fold_errors)

        # Append the errors from using the mean as the prediction
        cv_errors = y_test - y_train.mean()
        cvm_errors += list(cv_errors)

    # Convert error lists to np.array for vectorized calculations
    errors = np.array(errors)
    cvm_errors = np.array(cvm_errors)

    # Calculate common error metrics based on the recorded errors
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    tss = np.mean(cvm_errors**2)
    cvr2 = 1 - mse/tss
    if show:
        print("MAE: %.4f, MSE: %.4f, MSE-Mean: %.4f, CVR2: %.4f" % (mae, mse, tss, cvr2))
    return pd.Series(data=[mae, mse, cvr2], index=["MAE", "MSE", "CVR2"])


def create_combined_df(df_list, feature_set_list, response_col):
    """Combine arbitrary number of dataframes of the study form resulting from preprocess_df
        Used for combining the REFLECT and LEADS datasets (of mismatching columns)"""
    # Combine all features to get list of full features
    full_features = {"group_index", response_col}
    for fs in feature_set_list:
        full_features = full_features.union(fs)

    # Concatenate the rows of each dataframe to the full dataframe, will add NaNs for not found columns
    return_df = pd.DataFrame(columns=full_features)
    for i, df in enumerate(df_list):
        df["group_index"] = i
        for col in full_features:
            if col not in df.columns:
                df[col] = np.random.normal(loc=0, scale=0.001, size=df.shape[0])
        return_df = pd.concat([return_df, df.loc[:, full_features]], axis=0)

    # Reindex and fill the NaNs with mean (should be 0 for standardized dataframes)
    return_df.index = list(range(return_df.shape[0]))
    return_df.fillna(return_df.mean(), inplace=True)
    return return_df

if __name__ == "__main__":
    # Defining data paths for reading and writing
    directory = "C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/REFLECT/"
    reflect_data_filename = directory + "ActivitySummaryTransfer-REFLECT.csv"
    leads_data_filename = directory + "ActivitySummaryTransfer-LEADS.csv"
    sample_directory = "C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/REFLECT/saved_samples/"

    # Reading in the data
    reflect_df = pd.read_csv(reflect_data_filename)
    leads_df = pd.read_csv(leads_data_filename)

    # Defining arguments for future functions
    show = True
    response_col = "Post-IMI-Interest-Enjoyment"
    reflect_add_cols = ['C-DP-Prompts', 'C-DP-Posters', 'MeanHowIsItGoingLikert']
    # feature_cols_actions = ['MysterySolved', 'Duration', 'C-A-Conversation', 'C-A-BooksAndArticles', 'C-A-Worksheet',
    # 'C-A-PlotPoint', 'C-A-Scanner', 'C-A-WorksheetSubmit']
    feature_cols_props = ['FinalGameScore', 'MysterySolved', 'C-DP-Conversation', 'C-DP-BooksAndArticles',
                          'C-DP-Worksheet', 'C-DP-Scanner', 'C-A-PlotPoint', 'C-A-WorksheetSubmit']
    feature_cols = feature_cols_props
    scaling = True


    # Data preprocessing
    leads_df = leads_df.loc[leads_df["Condition"] == 1, :]
    leads_df, leads_scaler = preprocess_df(leads_df, feature_cols, scale=scaling)
    reflect_df, reflect_scaler = preprocess_df(reflect_df, feature_cols + reflect_add_cols, scale=scaling)
    combined_df = create_combined_df(df_list=[reflect_df, leads_df],
                                     feature_set_list=[feature_cols + reflect_add_cols, feature_cols],
                                     response_col=response_col)
    print(combined_df.shape)
    # combined_df = pd.concat([reflect_df, leads_df])

    # Print demographic information commonly used in methods/study description section
    print_demographics(reflect_df, response=response_col, title_key="REFLECT")
    print_demographics(leads_df, response=response_col, title_key="LEADS")
    #print_demographics(combined_df, response=response_col, title_key="BOTH")

    # Fit OLS models on each task separately and combined for descriptive purposes
    print_descriptive_lm(combined_df.loc[combined_df['group_index'] == 0, :], feature_cols + reflect_add_cols, response_col)
    print_descriptive_lm(combined_df.loc[combined_df['group_index'] == 1, :], feature_cols + reflect_add_cols, response_col)
    print_descriptive_lm(combined_df, feature_cols + reflect_add_cols, response_col)

    # Fit L2/BLR models on each task separately and combined
    evaluate_cv_performance(X=reflect_df.loc[:, feature_cols + reflect_add_cols], y=reflect_df.loc[:, response_col],
                            model=LinearRegression(), folds=reflect_df.shape[0], show=show)
    evaluate_cv_performance(X=reflect_df.loc[:, feature_cols + reflect_add_cols], y=reflect_df.loc[:, response_col],
                            model=Ridge(alpha=1.0), folds=reflect_df.shape[0], show=show)
    evaluate_cv_performance(X=reflect_df.loc[:, feature_cols + reflect_add_cols], y=reflect_df.loc[:, response_col],
                            model=BayesianRidge(lambda_1=1.0, lambda_2=1.0), folds=reflect_df.shape[0], show=show)

    # Fit Bayesian Hierarchical Models using MCMC
    pooled_samples = tpmc.sample_pooled_model(X=combined_df.loc[:, feature_cols + reflect_add_cols],
                                              y=combined_df.loc[:, response_col],
                                              features=feature_cols + reflect_add_cols,
                                              save_filename=sample_directory+"pooled")
    individual_samples = tpmc.sample_alternative_individual_group_model(X=combined_df.loc[:, feature_cols + reflect_add_cols],
                                                            y=combined_df.loc[:, response_col],
                                                            group_indices=combined_df.loc[:,"group_index"].values,
                                                            features=feature_cols + reflect_add_cols,
                                                            save_filename=sample_directory+"individual")
    hierarchical_samples = tpmc.sample_hierarchical_model(X=combined_df.loc[:, feature_cols + reflect_add_cols],
                                                          y=combined_df.loc[:, response_col],
                                                          group_indices=combined_df.loc[:,"group_index"].values,
                                                          features=feature_cols + reflect_add_cols,
                                                          save_filename=sample_directory+"hierarchical")

    # Running the manual cross validation scripts for Bayesian Hierarchical Models using MCMC
    # pooled_sample_accuracy = tpmc.cv_pooled_model(X=combined_df.loc[:, feature_cols + reflect_add_cols],
    #                                               y=combined_df.loc[:, response_col],
    #                                               group_indices=combined_df.loc[:, "group_index"],
    #                                               features=feature_cols + reflect_add_cols,
    #                                               save_filename=sample_directory+"pooled_accuracy_SD10.csv")
    # individual_sample_accuracy = tpmc.cv_alternative_individual_group_model(X=combined_df.loc[:, feature_cols + reflect_add_cols],
    #                                                                         y=combined_df.loc[:, response_col],
    #                                                                         group_indices=combined_df.loc[:, "group_index"],
    #                                                                         features=feature_cols + reflect_add_cols,
    #                                                                         save_filename=sample_directory+"individual_accuracy_SD10.csv")
    # hierarchical_sample_accuracy = tpmc.cv_hierarchical_model(X=combined_df.loc[:, feature_cols + reflect_add_cols],
    #                                                           y=combined_df.loc[:, response_col],
    #                                                           group_indices=combined_df.loc[:,"group_index"],
    #                                                           features=feature_cols + reflect_add_cols,
    #                                                           save_filename=sample_directory+"hierarchical_accuracy_SD10.csv")

