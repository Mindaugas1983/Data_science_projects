import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
import elsis_data_science.algorithms as alg


def reduce_features(train_df, importance, feature_count):
    """
    This function extracts given number of most important features from given dataframe
    Parameters:
    train_df: dataframe with all possible features
    importance: feature importance from function "get_importances"
    feature_count: number of most important features that we want to extract
    """

    reduced_train_df = train_df.loc[:, importance[importance.index < feature_count]['feature']]
    return reduced_train_df


def get_importances(train, imp):
    """
    This function orders features from given dataframe by importance and also adds value of importance in output
    Parameters:
    train: dataframe with features
    imp: coefficients  of features from logistic regression
    """

    order = pd.DataFrame()
    value = pd.DataFrame()
    column = pd.DataFrame()
    important_features_dict = {}
    for x, i in enumerate(imp):
        important_features_dict[x] = np.abs(i)
    order['order'] = sorted(important_features_dict,
                            key=important_features_dict.get,
                            reverse=True)
    value['importance'] = imp
    column['feature'] = train.columns
    imp = order.set_index('order').join(value).join(column)
    imp = imp.reset_index(drop=True)
    return imp


def increasing_crossvalidation(cv_model, train_origin, target, imp):
    """
    This function at prints cross-validation metrics of model trained with increasing number of features ordered by
    importance. Beginning with one feature and increasing feature count by 1 Parameters: cv_model: model for
    producing cross-validation results train_origin: dataframe with all features target: series with model target
    imp: coefficients  of features from logistic regression
    """

    origin_importance = imp
    print("all_features_count:", len(origin_importance))
    print(" ")
    for i in range(len(origin_importance)):
        print("features_used:", i + 1)
        i = i + 1
        trn = reduce_features(train_origin, origin_importance, i)
        scores = cross_validate(cv_model, trn, target, cv=10, scoring=['roc_auc', 'neg_log_loss'],
                                return_train_score=True)
        alg.print_results(scores)
