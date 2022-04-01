import cairosvg as svg
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import io
import shap
import iml
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def save_plot(destination, plot, name):

    """
    This function saves plot to given directory with given name as png format
    Parameters:
    destination: path to directory where to save plot
    plot: plot variable which we want to save
    name: name of the file to be created
    """
    
    if os.path.exists(destination + "/" + name + ".svg"):
        os.remove(destination + "/" + name + ".svg")
    if os.path.exists(destination):
        plot.figure.savefig(destination + "/" + name + ".svg", format='svg')
        svg.svg2png(url=destination + "/" + name + ".svg", write_to=destination + "/" + name + ".png")
        os.remove(destination + "/" + name + ".svg")
    else:
        print("The folder does not exist")

        
def save_observation_list(data, path, name):

    """
    This function saves  statistics of binary target in json format for documentation
    Parameters:
    data: dataframe with column named "Target" for target statistics
    path: path to folder where json file should be saved
    name: name of the json file to be created
    """
    
    if os.path.exists(path):
        if os.path.exists(path + "/" + name + ".json"):
            os.remove(path + "/" + name + ".json")
        amounts = data["Target"].value_counts()
        observations = pd.DataFrame({"Target": ["BAD", "GOOD", "TOTAL"], "Total": [amounts[0], amounts[1], amounts[1] + amounts[0]],
                                     "Percentage": [round(amounts[0] / (amounts[1] + amounts[0]) * 100, 2), round(amounts[1]/(amounts[1] + amounts[0]) * 100, 2), "-"]})
        cols = ["Target", "Total", "Percentage"]
        observations2 = observations[cols]
        observations_json = observations2.to_json(orient='records')
        with io.open(path + "/" + name + '.json', 'w') as f:
            f.write(observations_json)
    else:
        print("The folder does not exist")
        
        
def generate_auc_chart(train, target, model, random_st, n_splits, is_nn=0):

    """
    This function saves  statistics of binary target in json format for documentation
    Parameters:
    data: dataframe with column named "Target" for target statistics
    path: path to folder where json file should be saved
    name: name of the json file to be created
    """
    
    # fig = plt.figure(figsize=(22, 22))
    fig2 = plt.figure(figsize=(22, 22))
    # axes = plt.subplot()
    axes2 = plt.subplot()
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_st)
    y_real = []
    y_proba = []
    # lab =  ''
    # lab2 = ''
    for i, (train_index, test_index) in enumerate(k_fold.split(train)):
        xtrain, xtest = train.iloc[train_index], train.iloc[test_index]
        ytrain, ytest = target.iloc[train_index], target.iloc[test_index]
        if is_nn == 1:
            trained_model = KerasClassifier(build_fn=model, epochs=50, batch_size=20, verbose=1)
            trained_model.fit(xtrain, ytrain)
        else:
            trained_model = model.fit(xtrain, ytrain)
        pred_proba = trained_model.predict_proba(xtest)
        predictions = pred_proba[:, 1]
        fpr, tpr, threshold = roc_curve(ytest, predictions)
        lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
        axes2.step(fpr, tpr, label=lab2)
        y_real.append(ytest)
        y_proba.append(predictions)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    # precision, recall, _ = precision_recall_curve(y_real, y_proba)
    fpr, tpr, threshold = roc_curve(y_real, y_proba)
    # lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    lab2 = 'Overall AUC=%.4f' % (auc(fpr, tpr))
    axes2.step(fpr, tpr, label=lab2, lw=2, color='black')
    axes2.set_xlabel('FPR', fontsize=25)
    axes2.set_ylabel('TPR', fontsize=25)
    axes2.legend(loc='lower right', fontsize='small')
    axes2.set_title('AUC curve (10 folds)', fontsize=40)
    fig2.add_subplot(axes2)
    return fig2


def ensure_not_numpy(x):

    """
    This function is aa aditional function to calculate_top_contributors function. It prevents from passing dats in incorect format
    Parameters:
    x: variable that needs to be checked if it is as required
    """
    
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, np.str):
        return str(x)
    elif isinstance(x, np.generic):
        return float(np.asscalar(x))
    else:
        return x


def calculate_top_contributors(shap_values, features=None, feature_names=None, use_abs=False, return_df=False,
                               n_features=5):

    """
    This function if return_df is True: returns a pandas dataframe, if return_df is False: returns a flattened list by name, effect, and value
    Adapted from the SHAP package for visualizing the contributions of features towards a prediction.
    https://github.com/slundberg/shap

    Parameters:
    shap_values: np.array of floats
    features: pandas.core.series.Series, the data with the values
    feature_names: list, all the feature names/ column names
    use_abs: bool, if True, will sort the data by the absolute value of the feature effect
    return_df: bool, if True, will return a pandas dataframe, else will return a list of feature, effect, value
    n_features: int, the number of features to report on. If it equals -1 it will return the entire dataframe
    """
    
    assert not type(shap_values) == list, "The shap_values arg looks looks multi output, try shap_values[i]."
    assert len(shap_values.shape) == 1, "Expected just one row. Please only submit one row at a time."

    shap_values = np.reshape(shap_values, (1, len(shap_values)))
    instance = iml.Instance(np.zeros((1, len(feature_names))), features)
    link = iml.links.convert_to_link('identity')
    if n_features == -1:
        limit = len(feature_names)
    else:
        limit = min(n_features, len(feature_names))

    # explanation obj
    expl = iml.explanations.AdditiveExplanation(
        shap_values[0, limit-1],                 # base value
        np.sum(shap_values[0, :]),          # this row's prediction value
        shap_values[0, : limit],                # matrix
        None,
        instance,                           # <iml.common.Instance object >
        link,                               # 'identity'
        iml.Model(None, ["output value"]),  # <iml.common.Model object >
        iml.datatypes.DenseData(np.zeros((1, len(feature_names))), list(feature_names))
    )

    # Get the name, effect and value for each feature, if there was an effect
    features_ = {}
    for i in range(len(expl.data.group_names)):
        # set_trace()
        if expl.effects[i] != 0:
            features_[i] = {
                "effect": ensure_not_numpy(expl.effects[i]),
                "value": ensure_not_numpy(expl.instance.group_display_values[i]),
                "name": expl.data.group_names[i]
            }

    effect_df = pd.DataFrame([v for k, v in features_.items()])

    if use_abs:  # get the absolute value of effect
        effect_df['abs_effect'] = effect_df['effect'].apply(np.abs)
        effect_df.sort_values('abs_effect', ascending=False, inplace=True)
    else:
        effect_df.sort_values('effect', ascending=False, inplace=True)

    if not n_features == -1:
        effect_df = effect_df.head(n_features)
    if return_df:
        return effect_df.reset_index(drop=True)
    else:
        list_of_info = list(zip(effect_df.name, effect_df.effect, effect_df.value))
        effect_list = list(sum(list_of_info, ()))  # flattens the list of tuples
        return effect_list


def get_nn_feature_effects(test_x, test_app, n_features, shap_explainer, id_column='application_id'):

    """
    This function returns list of feature effects according shap model
    Parameters:
    test_X: dataframe of feature values
    test_app: Series with some identification column of observations (ssn , application_id or something similar)
    n_features: number of most important features which we want to analyze. if n_features = -1 then all features will be selected
    shap_explainer: trained shap model for providing feature importances
    id_column: name of a column for storing client identifier
    """
    
    results = pd.DataFrame(columns=['effect', 'name', 'value', 'abs_effect', 'application_id'])
    for i in range(len(test_x)):
        j = i + 1
        shap_values = shap_explainer.shap_values(test_x[i:j].values.astype('float'))
        if n_features == -1:
            cnt = len(test_x.columns)
        else:
            cnt = n_features
        first_row_effects_list = calculate_top_contributors(shap_values=np.asarray(shap_values).reshape(cnt), features=test_x.iloc[i],
                                                            feature_names=list(test_x.columns), return_df=True, n_features=n_features, use_abs=True)
        first_row_effects_list[id_column] = test_app[i]
        results = results.append(first_row_effects_list, ignore_index=True)
    return results


def save_table(data, path, name):

    """
    This function saves given dataframe in json format for documentation
    Parameters:
    data: dataframe that needs to be stored as json
    path: path to folder where json files needs to be stored
    name: name of json file
    """
    
    if os.path.exists(path):
        if os.path.exists(path + "/" + name + ".json"):
            os.remove(path + "/" + name + ".json")
        percentile_table = data.to_json(orient='records')
        with io.open(path + "/" + name + '.json', 'w') as f:
            f.write(percentile_table)
    else:
        print("The folder does not exist")


def model_performance_table(df, df_ts, metric='auc'):

    """
    This function returns a dataframe with main model metrics for documentation purposes
    Parameters:
    data: dataframe with model cross-validation results
    data_ts: dataframe with model time-series cross-validation results
    metric: type of metric - if matric = 'rmse' resulting table will be based on RMSE metric, if matric = 'logloss' resulting table will be based on Log loss metric, otherwise AUC metric will be used
    """
    
    if metric == 'rmse':
        results = pd.DataFrame({"RMSE table": ["cross-validation resuts (10 folds)", " time series cross-validation resuts (10 folds)", "Last fold of time series cross_validation"]
                            , "Train RMSE": [np.average(df["train_neg_mean_squared_error"]), np.average(df_ts["train_neg_mean_squared_error"]), df_ts["train_neg_mean_squared_error"][9]]
                            , "Train std": [np.std(df["train_neg_mean_squared_error"]), np.std(df_ts["train_neg_mean_squared_error"]), "X"]
                            , "Test RMSE": [np.average(df["test_neg_mean_squared_error"]), np.average(df_ts["test_neg_mean_squared_error"]), df_ts["test_neg_mean_squared_error"][9]]
                            , "Test std": [np.std(df["test_neg_mean_squared_error"]), np.std(df_ts["test_neg_mean_squared_error"]), "X"]
                            })
    elif metric == 'logloss':
        results = pd.DataFrame({"Logloss table": ["cross-validation resuts (10 folds)", " time series cross-validation resuts (10 folds)", "Last fold of time series cross_validation"]
                            , "Train Log loss": [np.average(df["train_neg_log_loss"]) , np.average(df_ts["train_neg_log_loss"]), df_ts["train_neg_log_loss"][9]]
                            , "Train std": [np.std(df["train_neg_log_loss"]), np.std(df_ts["train_neg_log_loss"]), "X"]
                            , "Test Log loss": [np.average(df["test_neg_log_loss"]), np.average(df_ts["test_neg_log_loss"]), df_ts["test_neg_log_loss"][9]]
                            , "Test std": [np.std(df["test_neg_log_loss"]), np.std(df_ts["test_neg_log_loss"]), "X"]
                            })
    else:
         results = pd.DataFrame({"AUC table": ["cross-validation resuts (10 folds)", " time series cross-validation resuts (10 folds)", "Last fold of time series cross_validation"]
                            , "Train AUC": [np.average(df["train_roc_auc"]), np.average(df_ts["train_roc_auc"]), df_ts["train_roc_auc"][9]]
                            , "Train std": [np.std(df["train_roc_auc"]), np.std(df_ts["train_roc_auc"]), "X"]
                            , "Test AUC": [np.average(df["test_roc_auc"]), np.average(df_ts["test_roc_auc"]), df_ts["test_roc_auc"][9]]
                            , "Test std": [np.std(df["test_roc_auc"]), np.std(df_ts["test_roc_auc"]), "X"]
                            })
        
    return results


def percentile_bins(data, nr):

    """
    This function returns a dataframe of limits of percentile bins in columns Start and End

    Parameters:
    data - is expected to be Series of data for which percentile bins should be calculated
    nr - is number of bins needs to be generated in output table
    """
    
    arr = []
    i = 0
    inc = int(100 / nr)
    i += inc
    while i < 100:
        val = np.percentile(data, i)
        arr.append(val)
        i += inc
    arr.insert(0, np.percentile(data, 0))
    arr.append(np.percentile(data, 100))
    arr2 = pd.DataFrame(arr)
    arr2['Nr'] = arr2.index
    arr2['Nr2'] = arr2.index + 1
    arr3 = arr2.set_index('Nr').join(arr2.set_index('Nr2'), how='inner', lsuffix='_B', rsuffix='_A')
    arr4 = pd.DataFrame()
    arr4['Start'] = arr3['0_A']
    arr4['End'] = arr3['0_B']
    return arr4


def assign_bin(data, column, result, pct_bins_df):

    """
    This function adds a column to given dataframe in which there is assigned bin for each prediction

    Parameters:
    data - dataframe  in which is possible to find prediction column
    column - column name of prediction column
    result - column name of results column to be created
    pct_bins_df - dataframe of limits of percentile bins created with percentile_bins function
    """
    
    pct_bins_df2 = pct_bins_df
    pct_bins_df2['Nr'] = pct_bins_df2.index

    def get_bin(var, df):
        # pdb.set_trace()
        res_df = df[(df['Start'] <= var) & (df['End'] >= var)]
        res_df = res_df.reset_index(drop=True)
        # pdb.set_trace()
        result2 = res_df['Nr']
        # pdb.set_trace()
        return result2[0]
    # pdb.set_trace()
    data[result] = data.apply(lambda x: get_bin(x[column], pct_bins_df2), axis=1)
    return data


def plot_auc_comparison(y, champion_values, challenger_values):
    """
    This function plots two AUC curves in one chart for comparison

    Parameters:
    y - dataframe  in which is possible to find prediction column
    champion_values - dataframe with first set of predictions for comparison
    challenger_values - dataframe with second set of predictions for comparison
    """
    # fig2 = plt.figure(figsize=(12, 12))
    axes = plt.subplot()
    axes2 = plt.subplot()
    # axes3 = plt.subplot()
    fpr, tpr, threshold = roc_curve(y, challenger_values)
    fpr3, tpr3, threshold3 = roc_curve(y, champion_values)
    lab2 = 'Challenger AUC=%.4f' % (auc(fpr, tpr))
    lab3 = 'Champion AUC=%.4f' % (auc(fpr3, tpr3))
    # axes2.step(fpr, tpr, label=lab2)
    axes.plot([0, 1], lw=2, color='red')
    axes2.step(fpr, tpr, label=lab2, lw=2, color='blue')
    axes2.step(fpr3, tpr3, label=lab3, lw=2, color='green')
    axes2.set_xlabel('FPR')
    axes2.set_ylabel('TPR')
    axes2.legend(loc='lower left', fontsize='small')
    axes2.set_title('Comparison of ROC curves', fontsize=16)
