import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import pickle
from sklearn.model_selection import KFold


SAMPLE_NUMBER = 5000
HYPER_SD = 10
CV_SAMPLES = 1000


def save_samples(save_filename, samples):
    if save_filename != "":
        save_filename += "_%d.pkl" % SAMPLE_NUMBER
        with open(save_filename, 'wb') as buff:
            pickle.dump(samples, buff)


def sample_pooled_model(X, y, features, save_filename=""):
    """Use MCMC to sample from model parameters from pooling all of the data together (pooling done outside)"""
    with pm.Model() as pooled_model:
        # Intercept for predicting response y using normal distribution with weak prior
        alpha = pm.Normal('alpha', mu=0, sd=HYPER_SD)

        # Vector of parameters to create linear combination of inputs x to predict y
        beta = pm.Normal('beta', mu=0, sd=HYPER_SD, shape=len(features))

        # The prediction of y (mean) is a linear combination of parameters and inputs
        y_pred = alpha + beta.dot(X.T)

        sigma = pm.HalfCauchy('sigma', 5)

        # The sampling distribution of the response is assumed to have normally distributed errors around the prediction
        y_like = pm.Normal('y_like', mu=y_pred, sd=sigma, observed=y)

    with pooled_model:
        pooled_samples = pm.sample(SAMPLE_NUMBER)

    save_samples(save_filename=save_filename, samples=pooled_samples)

    return pooled_samples


def cv_pooled_model(X, y, group_indices, features, folds=10, save_filename=""):
    """Use MCMC to sample from model parameters from pooling all of the data together (pooling done outside)"""
    kf = KFold(n_splits=folds, shuffle=True)
    accuracy_dict = dict()
    fold_number = 0
    for train_index, test_index in kf.split(X):
        X_train, y_train = X.loc[train_index, :], y.loc[train_index]
        group_indices_train = group_indices.loc[train_index]
        with pm.Model() as pooled_model:
            # Intercept for predicting response y using normal distribution with weak prior
            alpha = pm.Normal('alpha', mu=0, sd=HYPER_SD)

            # Vector of parameters to create linear combination of inputs x to predict y
            beta = pm.Normal('beta', mu=0, sd=HYPER_SD, shape=len(features))

            # The prediction of y (mean) is a linear combination of parameters and inputs
            y_pred = alpha + beta.dot(X_train.T)

            sigma = pm.HalfCauchy('sigma', 5)

            # The sampling distribution of the response is assumed to have normally distributed errors around the prediction
            y_like = pm.Normal('y_like', mu=y_pred, sd=sigma, observed=y_train)

        with pooled_model:
            # Get samples from posterior of model without test data
            pooled_samples = pm.sample(CV_SAMPLES)

            # Get the test data
            X_test, y_test = X.loc[test_index, :], y.loc[test_index]
            group_indices_test = group_indices.loc[test_index]

            reflect_size = (group_indices_test == 0).sum()
            leads_size = (group_indices_test == 1).sum()
            reflect_preds = np.tile(pooled_samples['alpha'], (reflect_size, 1)).T + pooled_samples['beta'].dot(X_test.loc[group_indices_test == 0, :].T)
            leads_preds = np.tile(pooled_samples['alpha'], (leads_size, 1)).T + pooled_samples['beta'].dot(X_test.loc[group_indices_test == 1, :].T)
            all_preds = np.concatenate([reflect_preds, leads_preds], axis=1)

            reflect_accuracy = fold_group_accuracy_metrics(predictions=reflect_preds,
                                                           y_test=y_test.loc[group_indices_test == 0],
                                                           train_mean=y_train.loc[group_indices_train == 0].mean(),
                                                           name_key_prefix="R-")
            leads_accuracy = fold_group_accuracy_metrics(predictions=leads_preds,
                                                         y_test=y_test.loc[group_indices_test == 1],
                                                         train_mean=y_train.loc[group_indices_train == 1].mean(),
                                                         name_key_prefix="L-")
            all_accuracy = fold_group_accuracy_metrics(predictions=all_preds,
                                                       y_test=np.concatenate([y_test.loc[group_indices_test == 0],
                                                                              y_test.loc[group_indices_test == 1]]),
                                                       train_mean=y_train.mean(),
                                                       name_key_prefix="All-")
            accuracy_dict["Fold-%d" % fold_number] = pd.concat([reflect_accuracy, leads_accuracy, all_accuracy])
            fold_number += 1

    # Reformat accuracy results and output if save_filename given
    accuracy_data = pd.DataFrame.from_dict(accuracy_dict, orient='index')
    if save_filename != "":
        accuracy_data.to_csv(save_filename)

    return accuracy_data


def sample_individual_group_model(X, y, group_indices, features, save_filename=""):
    """FOR SOME REASON THIS METHODOLOGY DOES NOT WORK, IT IS GROUPING ALL OF THE DATA TOGETHER SIMILAR TO A POOLED MODEL
    USE THE ALTERNATIVE VERSION
    Use MCMC to sample from model parameters treating each group as completely separate (independent parameters)"""
    total_groups = len(np.unique(group_indices))
    with pm.Model() as individual_model:
        # Intercepts for each group
        alpha = pm.Normal('alpha', mu=0, sd=HYPER_SD, shape=total_groups)

        # Slopes for each group
        beta = pm.Normal('beta', mu=0, sd=HYPER_SD, shape=(total_groups, len(features)))

        # Model Error
        sigma_y = pm.HalfCauchy('sigma_y', 2, shape=total_groups)

        # Expected Value
        #y_hat = alpha[group_indices] + beta[group_indices, :].dot(X.T)
        y_hat = alpha[group_indices] + pm.math.dot(beta[group_indices], X.T)

        # Data Likelihood
        y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y[group_indices], observed=y)

    with individual_model:
        individual_samples = pm.sample(SAMPLE_NUMBER)

    save_samples(save_filename=save_filename, samples=individual_samples)

    return individual_samples


def sample_alternative_individual_group_model(X, y, group_indices, features, save_filename=""):
    """Use MCMC to sample from model parameters treating each group as completely separate (independent parameters)"""
    total_groups = len(np.unique(group_indices))
    with pm.Model() as individual_model:
        # Intercepts for each group
        alpha0 = pm.Normal('alpha_reflect', mu=0, sd=HYPER_SD)
        alpha1 = pm.Normal('alpha_leads', mu=0, sd=HYPER_SD)

        # Slopes for each group
        beta0 = pm.Normal('beta_reflect', mu=0, sd=HYPER_SD, shape=len(features))
        beta1 = pm.Normal('beta_leads', mu=0, sd=HYPER_SD, shape=len(features))

        # Model Error
        sigma_y0 = pm.HalfCauchy('sigma_y_reflect', 2)
        sigma_y1 = pm.HalfCauchy('sigma_y_leads', 2)

        # Expected Value
        y_hat0 = alpha0 + beta0.dot(X.loc[group_indices == 0, :].T)
        y_hat1 = alpha1 + beta1.dot(X.loc[group_indices == 1, :].T)

        # Data Likelihood
        y_like0 = pm.Normal('y_like_reflect', mu=y_hat0, sd=sigma_y0, observed=y[group_indices == 0])
        y_like1 = pm.Normal('y_like_leads', mu=y_hat1, sd=sigma_y1, observed=y[group_indices == 1])

    with individual_model:
        individual_samples = pm.sample(SAMPLE_NUMBER)

    save_samples(save_filename=save_filename, samples=individual_samples)

    return individual_samples


def fold_group_accuracy_metrics(predictions, y_test, train_mean, name_key_prefix=""):
    interval_size = len(y_test)
    mean_predictions = predictions.mean(axis=0)
    sorted_predictions = np.sort(predictions, axis=0)
    lower_predictions = sorted_predictions[int(0.05 * CV_SAMPLES), :]
    upper_predictions = sorted_predictions[int(0.95 * CV_SAMPLES), :]

    # Calculate error metrics for the test fold
    errors = y_test - mean_predictions
    in_pred_interval = np.logical_and(lower_predictions < y_test, y_test < upper_predictions)
    interval_mse = np.mean(errors ** 2)
    interval_mae = np.mean(np.abs(errors))
    interval_mean_mse = np.mean((y_test - train_mean) ** 2)
    interval_mean_mae = np.mean(np.abs(y_test - train_mean))

    # return data from the
    index_list = ["%s%s" % (name_key_prefix, e) for e in ["MSE", "MAE", "In-95-Interval", "Test Size", "MeanMSE", "MeanMAE"]]
    return pd.Series(
        data=[interval_mse, interval_mae, in_pred_interval.sum(), interval_size, interval_mean_mse, interval_mean_mae],
        index=index_list)


def cv_alternative_individual_group_model(X, y, group_indices, features, folds=10, save_filename=""):
    """Use MCMC to sample from model parameters treating each group as completely separate (independent parameters)"""
    kf = KFold(n_splits=folds, shuffle=True)
    accuracy_dict = dict()
    fold_number = 0
    for train_index, test_index in kf.split(X):
        X_train, y_train = X.loc[train_index, :], y.loc[train_index]
        group_indices_train = group_indices.loc[train_index]
        with pm.Model() as individual_model:
            # Intercepts for each group
            alpha0 = pm.Normal('alpha_reflect', mu=0, sd=HYPER_SD)
            alpha1 = pm.Normal('alpha_leads', mu=0, sd=HYPER_SD)

            # Slopes for each group
            beta0 = pm.Normal('beta_reflect', mu=0, sd=HYPER_SD, shape=len(features))
            beta1 = pm.Normal('beta_leads', mu=0, sd=HYPER_SD, shape=len(features))

            # Model Error
            sigma_y0 = pm.HalfCauchy('sigma_y_reflect', 2)
            sigma_y1 = pm.HalfCauchy('sigma_y_leads', 2)

            # Expected Value
            y_hat0 = alpha0 + beta0.dot(X_train.loc[group_indices_train == 0, :].T)
            y_hat1 = alpha1 + beta1.dot(X_train.loc[group_indices_train == 1, :].T)

            # Data Likelihood
            y_like0 = pm.Normal('y_like_reflect', mu=y_hat0, sd=sigma_y0, observed=y_train[group_indices_train == 0])
            y_like1 = pm.Normal('y_like_leads', mu=y_hat1, sd=sigma_y1, observed=y_train[group_indices_train == 1])

        with individual_model:
            individual_samples = pm.sample(CV_SAMPLES)
            X_test, y_test = X.loc[test_index, :], y.loc[test_index]
            group_indices_test = group_indices.loc[test_index]

            reflect_size = (group_indices_test == 0).sum()
            leads_size = (group_indices_test == 1).sum()
            reflect_preds = np.tile(individual_samples['alpha_reflect'], (reflect_size, 1)).T + individual_samples['beta_reflect'].dot(X_test.loc[group_indices_test == 0, :].T)
            leads_preds = np.tile(individual_samples['alpha_leads'], (leads_size, 1)).T + individual_samples['beta_leads'].dot(X_test.loc[group_indices_test == 1, :].T)
            all_preds = np.concatenate([reflect_preds, leads_preds], axis=1)

            reflect_accuracy = fold_group_accuracy_metrics(predictions=reflect_preds,
                                                           y_test=y_test.loc[group_indices_test == 0],
                                                           train_mean=y_train.loc[group_indices_train == 0].mean(),
                                                           name_key_prefix="R-")
            leads_accuracy = fold_group_accuracy_metrics(predictions=leads_preds,
                                                         y_test=y_test.loc[group_indices_test == 1],
                                                         train_mean=y_train.loc[group_indices_train == 1].mean(),
                                                         name_key_prefix="L-")
            all_accuracy = fold_group_accuracy_metrics(predictions=all_preds,
                                                       y_test=np.concatenate([y_test.loc[group_indices_test == 0], y_test.loc[group_indices_test == 1]]),
                                                       train_mean=y_train.mean(),
                                                       name_key_prefix="All-")
            accuracy_dict["Fold-%d" % fold_number] = pd.concat([reflect_accuracy, leads_accuracy, all_accuracy])

        fold_number += 1

    accuracy_data = pd.DataFrame.from_dict(accuracy_dict, orient='index')
    if save_filename != "":
        accuracy_data.to_csv(save_filename)

    return accuracy_data


def sample_hierarchical_model(X, y, group_indices, features, save_filename=""):
    """Currently manually performing the groups since the vectorized information is somehow sharing information"""
    total_groups = len(np.unique(group_indices))
    with pm.Model() as hierarchical_model:
        # Hyperpriors that act as linking variables for groups
        mu_alpha = pm.Normal('mu_alpha', mu=0., sd=HYPER_SD)
        sigma_alpha = pm.HalfCauchy('sigma_alpha', 2)

        mu_beta = pm.Normal('mu_beta', mu=0, sd=HYPER_SD, shape=len(features))
        sigma_beta = pm.HalfCauchy('sigma_beta', 2, shape=len(features))

        # Random Intercepts for groups
        alpha_leads = pm.Normal('alpha_reflect', mu=mu_alpha, sd=sigma_alpha)
        alpha_reflect = pm.Normal('alpha_leads', mu=mu_alpha, sd=sigma_alpha)

        # Random Slopes for groups
        beta_leads = pm.Normal('beta_reflect', mu=mu_beta, sd=sigma_beta, shape=len(features))
        beta_reflect = pm.Normal('beta_leads', mu=mu_beta, sd=sigma_beta, shape=len(features))

        # Model error
        sigma_y_leads = pm.HalfCauchy('sigma_y_reflect', 2)
        sigma_y_reflect = pm.HalfCauchy('sigma_y_leads', 2)

        # Expected Value
        y_hat_leads = alpha_leads + beta_leads.dot(X.loc[group_indices == 0, :].T)
        y_hat_reflect = alpha_reflect + beta_reflect.dot(X.loc[group_indices == 1, :].T)

        # Data Likelihood
        y_like_leads = pm.Normal('y_like_reflect', mu=y_hat_leads, sd=sigma_y_leads, observed=y[group_indices == 0])
        y_like_reflect = pm.Normal('y_like_leads', mu=y_hat_reflect, sd=sigma_y_reflect, observed=y[group_indices == 1])

    with hierarchical_model:
        hierarchical_samples = pm.sample(SAMPLE_NUMBER)

    save_samples(save_filename=save_filename, samples=hierarchical_samples)

    return hierarchical_samples


def cv_hierarchical_model(X, y, group_indices, features, folds=10, save_filename=""):
    """Currently manually performing the groups since the vectorized information is somehow sharing information"""
    kf = KFold(n_splits=folds, shuffle=True)
    accuracy_dict = dict()
    fold_number = 0
    for train_index, test_index in kf.split(X):
        X_train, y_train = X.loc[train_index, :], y.loc[train_index]
        group_indices_train = group_indices.loc[train_index]
        with pm.Model() as hierarchical_model:
            # Hyperpriors that act as linking variables for groups
            mu_alpha = pm.Normal('mu_alpha', mu=0., sd=HYPER_SD)
            sigma_alpha = pm.HalfCauchy('sigma_alpha', 2)

            mu_beta = pm.Normal('mu_beta', mu=0, sd=HYPER_SD, shape=len(features))
            sigma_beta = pm.HalfCauchy('sigma_beta', 2, shape=len(features))

            # Random Intercepts for groups
            alpha_leads = pm.Normal('alpha_reflect', mu=mu_alpha, sd=sigma_alpha)
            alpha_reflect = pm.Normal('alpha_leads', mu=mu_alpha, sd=sigma_alpha)

            # Random Slopes for groups
            beta_leads = pm.Normal('beta_reflect', mu=mu_beta, sd=sigma_beta, shape=len(features))
            beta_reflect = pm.Normal('beta_leads', mu=mu_beta, sd=sigma_beta, shape=len(features))

            # Model error
            sigma_y_leads = pm.HalfCauchy('sigma_y_reflect', 2)
            sigma_y_reflect = pm.HalfCauchy('sigma_y_leads', 2)

            # Expected Value
            y_hat_leads = alpha_leads + beta_leads.dot(X.loc[group_indices == 0, :].T)
            y_hat_reflect = alpha_reflect + beta_reflect.dot(X.loc[group_indices == 1, :].T)

            # Data Likelihood
            y_like_leads = pm.Normal('y_like_reflect', mu=y_hat_leads, sd=sigma_y_leads, observed=y[group_indices == 0])
            y_like_reflect = pm.Normal('y_like_leads', mu=y_hat_reflect, sd=sigma_y_reflect, observed=y[group_indices == 1])

        with hierarchical_model:
            hierarchical_samples = pm.sample(CV_SAMPLES)
            X_test, y_test = X.loc[test_index, :], y.loc[test_index]
            group_indices_test = group_indices.loc[test_index]

            reflect_size = (group_indices_test == 0).sum()
            leads_size = (group_indices_test == 1).sum()
            reflect_preds = np.tile(hierarchical_samples['alpha_reflect'], (reflect_size, 1)).T + hierarchical_samples[
                'beta_reflect'].dot(X_test.loc[group_indices_test == 0, :].T)
            leads_preds = np.tile(hierarchical_samples['alpha_leads'], (leads_size, 1)).T + hierarchical_samples[
                'beta_leads'].dot(X_test.loc[group_indices_test == 1, :].T)
            all_preds = np.concatenate([reflect_preds, leads_preds], axis=1)

            reflect_accuracy = fold_group_accuracy_metrics(predictions=reflect_preds,
                                                           y_test=y_test.loc[group_indices_test == 0],
                                                           train_mean=y_train.loc[group_indices_train == 0].mean(),
                                                           name_key_prefix="R-")
            leads_accuracy = fold_group_accuracy_metrics(predictions=leads_preds,
                                                         y_test=y_test.loc[group_indices_test == 1],
                                                         train_mean=y_train.loc[group_indices_train == 1].mean(),
                                                         name_key_prefix="L-")
            all_accuracy = fold_group_accuracy_metrics(predictions=all_preds,
                                                       y_test=np.concatenate([y_test.loc[group_indices_test == 0],
                                                                              y_test.loc[group_indices_test == 1]]),
                                                       train_mean=y_train.mean(),
                                                       name_key_prefix="All-")
            accuracy_dict["Fold-%d" % fold_number] = pd.concat([reflect_accuracy, leads_accuracy, all_accuracy])

        fold_number += 1

    accuracy_data = pd.DataFrame.from_dict(accuracy_dict, orient='index')
    if save_filename != "":
        accuracy_data.to_csv(save_filename)

    return accuracy_data

