import pandas as pd
# from sklearn.model_selection import StratifiedShuffleSplit
# from xgboost import XGBClassifier
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
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


def decreasing_crossvalidation(origin_model, cv_model, train_origin, target, itype='weight'):

    """

    This function at prints cross-validation metrics of  xgb model trained with increasing number of features ordered
    by importance. Beginning with one feature and increasing feature count by 1 Parameters: origin_model: xgb model
    which was used to extract feature importance cv_model: model for producing cross-validation results train_origin:
    dataframe with all features target: series with model target itype: xgboost feature importance type - 'weight',
    'gain' or 'cover'

    """
    
    origin_importance = pd.DataFrame(list(origin_model.get_booster().get_score(importance_type=itype).items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
    origin_importance.reset_index(drop=True, inplace=True)
    print("Important_features_count:", len(origin_importance))
    print(" ")
    for i in range(len(origin_importance)):
        i = i + 1
        trn = reduce_features(train_origin, origin_importance, i)
        scores = cross_validate(cv_model, trn, target, cv=10, scoring=['roc_auc', 'neg_log_loss'], return_train_score=True)   # TimeSeriesSplit(10).split(trn)
        print("XGB feature count: ", i)
        alg.print_results(scores)
