"""Class definition for an error structure for storing complicated error models and evaluating/plotting the metrics"""
import pandas as pd
import numpy as np


class ErrorStruct:
    """Will contain a list of error lists for each (model, feature_set), 
    with each error list being one student's LOOCV errors, and the list of these errors being all folds errors"""
    def __init__(self, name, baseline_keyword, models, keywords):
        self.name = name
        self.errors = dict()
        self.min_predictions = dict()
        self.max_predictions = dict()

        self.baseline_model = baseline_keyword
        self.models = list(models)
        self.keywords = list(keywords)

        self.errors[baseline_keyword] = []
        self.min_predictions[baseline_keyword] = np.inf
        self.max_predictions[baseline_keyword] = 0

        if len(keywords) > 0:  # Indicates there are keywords for each model (different feature subsets)
            for model in models:
                self.errors[model] = dict()
                self.min_predictions[model] = dict()
                self.max_predictions[model] = dict()
                for keyword in keywords:
                    self.errors[model][keyword] = []
                    self.min_predictions[model][keyword] = np.inf
                    self.max_predictions[model][keyword] = 0
        else:  # Indicates there are no subgroups per model
            for model in models:
                self.errors[model] = dict()
                self.min_predictions[model] = np.inf
                self.max_predictions[model] = 0

    def add_error_list(self, error_list, model, keyword=""):
        """Add a fold's errors (from cross-validation) to appropriate model and feature set"""
        number_errors = len(error_list)
        if len(keyword) > 1:
            self.errors[model][keyword].append(error_list)

            if number_errors < self.min_predictions[model][keyword]:
                self.min_predictions[model][keyword] = number_errors
            if number_errors > self.max_predictions[model][keyword]:
                self.max_predictions[model][keyword] = number_errors
        else:
            self.errors[model].append(error_list)
            if number_errors < self.min_predictions[model]:
                self.min_predictions[model] = number_errors
            if number_errors > self.max_predictions[model]:
                self.max_predictions[model] = number_errors

    def output_diagnostics(self):
        """Output basic summary statistics of the errors added to the object"""
        print("Minimum amount of predictions:%d for Students:%d in Baseline:%s" %
              (self.min_predictions[self.baseline_model], len(self.errors[self.baseline_model]), self.baseline_model))
        for model in self.models:
            for keyword in self.keywords:
                print("Minimum amount of predictions:%d for Students:%d in Model:%s with Features:%s" %
                      (self.min_predictions[model][keyword], len(self.errors[model][keyword]), model, keyword))

    def error_by_interval(self, model, keyword="", method="min"):
        """Calculate the average absolute error over each prediction (time) interval and return series"""
        if len(keyword) > 1:
            full_error_df = pd.DataFrame(self.errors[model][keyword])
        else:
            full_error_df = pd.DataFrame(self.errors[model])
        error_absolute_means = np.abs(full_error_df).mean(axis=0)
        error_stds = np.abs(full_error_df).std(axis=0)
        return error_absolute_means, error_stds

    def error_last_interval_per_fold(self, model, keyword=""):
        """Get the last prediction per cv fold (one student) and return the MAE/MSE of these |students| predictions"""
        if len(keyword) > 1:
            full_errors = self.errors[model][keyword]
        else:
            full_errors = self.errors[model]

        last_interval_errors = []
        for subject_error_list in full_errors:
            last_interval_errors.append(subject_error_list[-1])
        array_errors = np.array(last_interval_errors)
        mae = np.mean(np.abs(array_errors))
        std_mae = np.std(np.abs(array_errors))
        mse = np.mean(array_errors**2)
        std_mse = np.std(array_errors**2)
        overall_mae = self.all_error(model=model, keyword=keyword)
        return mae, std_mae, mse, std_mse, overall_mae

    def all_error(self, model, keyword=""):
        if len(keyword) > 1:
            full_errors = self.errors[model][keyword]
        else:
            full_errors = self.errors[model]
        count = 0
        error_sum = 0
        for subject_error_list in full_errors:
            count += len(subject_error_list)
            error_sum += np.sum(np.abs(np.array(subject_error_list)))
        overall_mae = error_sum / count
        return overall_mae




