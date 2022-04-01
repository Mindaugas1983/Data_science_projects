import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import scipy.integrate as integrate
import time
from functools import wraps


def print_results(scores, metrics=2):
    """
    This function prints results cross-validation function in convenient format
    Parameters:
    scores: output of cross-validation function
    metrics: if metrics = 1 then prints Log loss metrics, if metrics = 2 prints both Log loss and AUC metrics, if metrics = 3 prints root mean squared error metrics
    """

    if metrics == 1:
        print("Log_loss")
        print("average test score: ", np.average(scores["test_score"]), " average train score: ",
              np.average(scores["train_score"]))
        print("minimum test score: ", np.min(scores["test_score"]), "minimum train score: ",
              np.min(scores["train_score"]))
        print("maximum test score: ", np.max(scores["test_score"]), "maximum train score: ",
              np.max(scores["train_score"]))
        print("std test score: ", np.std(scores["test_score"]), " std train score: ", np.std(scores["train_score"]))
        x = np.max(scores["test_score"]) - np.min(scores["test_score"])
        x2 = np.max(scores["train_score"]) - np.min(scores["train_score"])
        print("biggest difference in test: ", x, "biggest difference in train: ", x2)
        print("test scores: ", scores["test_score"])
        print("train scores: ", scores["train_score"])
    elif metrics == 3:
        print("RMSE")
        print("average test score: ", np.average(scores["test_neg_mean_squared_error"]), " average train score: ",
              np.average(scores["train_neg_mean_squared_error"]))
        print("minimum test score: ", np.min(scores["test_neg_mean_squared_error"]), "minimum train score: ",
              np.min(scores["train_neg_mean_squared_error"]))
        print("maximum test score: ", np.max(scores["test_neg_mean_squared_error"]), "maximum train score: ",
              np.max(scores["train_neg_mean_squared_error"]))
        print("std test score: ", np.std(scores["test_neg_mean_squared_error"]), " std train score: ",
              np.std(scores["train_neg_mean_squared_error"]))
        x = np.max(scores["test_neg_mean_squared_error"]) - np.min(scores["test_neg_mean_squared_error"])
        x2 = np.max(scores["train_neg_mean_squared_error"]) - np.min(scores["train_neg_mean_squared_error"])
        print("biggest difference in test: ", x, "biggest difference in train: ", x2)
        print("test scores: ", scores["test_neg_mean_squared_error"])
        print("train scores: ", scores["train_neg_mean_squared_error"])
    else:
        print("AUC")
        print("average test score: ", np.average(scores["test_roc_auc"]), " average train score: ",
              np.average(scores["train_roc_auc"]))
        print("average test gini score: ", np.average(scores["test_roc_auc"] * 2 - 1), " average train gini score: ",
              np.average(scores["train_roc_auc"] * 2 - 1))
        print("minimum test score: ", np.min(scores["test_roc_auc"]), "minimum train score: ",
              np.min(scores["train_roc_auc"]))
        print("maximum test score: ", np.max(scores["test_roc_auc"]), "maximum train score: ",
              np.max(scores["train_roc_auc"]))
        print("std test score: ", np.std(scores["test_roc_auc"]), " std train score: ",
              np.std(scores["train_roc_auc"]))
        print("std gini test score: ", np.std(scores["test_roc_auc"] * 2 - 1), " std gini train score: ",
              np.std(scores["train_roc_auc"] * 2 - 1))
        x = np.max(scores["test_roc_auc"]) - np.min(scores["test_roc_auc"])
        x2 = np.max(scores["train_roc_auc"]) - np.min(scores["train_roc_auc"])
        print("biggest difference in test: ", x, "biggest difference in train: ", x2)
        print("test scores: ", scores["test_roc_auc"])
        print("train scores: ", scores["train_roc_auc"])
        print(" ")
        print("Log_loss")
        print("average test score: ", np.average(scores["test_neg_log_loss"]), " average train score: ",
              np.average(scores["train_neg_log_loss"]))
        print("minimum test score: ", np.min(scores["test_neg_log_loss"]), "minimum train score: ",
              np.min(scores["train_neg_log_loss"]))
        print("maximum test score: ", np.max(scores["test_neg_log_loss"]), "maximum train score: ",
              np.max(scores["train_neg_log_loss"]))
        print("std test score: ", np.std(scores["test_neg_log_loss"]), " std train score: ",
              np.std(scores["train_neg_log_loss"]))
        x = np.max(scores["test_neg_log_loss"]) - np.min(scores["test_neg_log_loss"])
        x2 = np.max(scores["train_neg_log_loss"]) - np.min(scores["train_neg_log_loss"])
        print("biggest difference in test: ", x, "biggest difference in train: ", x2)
        print("test scores: ", scores["test_neg_log_loss"])
        print("train scores: ", scores["train_neg_log_loss"])


def get_array_of_class_weights(y, weight0, weight1):
    """
    This function returns numpy array of weights for binary target
    Parameters:
    y: series of binary target values
    weight0: weight for target 0
    weight: weight for target 1
    """
    w_array = np.ones(y.shape[0], dtype='float')
    for i, val in enumerate(y):
        w_array[i] = np.where(val == 0, weight0, weight1)
    return w_array


def plot_roc_curve(x_size, y_size, target, res):
    """
    This function plots ROC curve from outputs of predict function and target
    Parameters:
    x_size: width of plot
    y_size: height of plot
    target: series with target values
    res: outputs of xgboost predict function
    """

    fig = plt.figure(figsize=(x_size, y_size))
    axes = plt.subplot()
    axes2 = plt.subplot()
    fpr, tpr, threshold = roc_curve(target, pd.DataFrame(res))
    lab2 = ' AUC=%.4f' % (auc(fpr, tpr))
    axes.plot([0, 1], lw=2, color='red')
    axes2.step(fpr, tpr, label=lab2, lw=2, color='blue')
    axes2.set_xlabel('FPR')
    axes2.set_ylabel('TPR')
    axes2.legend(loc='lower left', fontsize='small')
    axes2.set_title('XGB test data ROC curve ', fontsize=16)
    fig.add_subplot(axes2)
    fig.add_subplot(axes)
    # axes2
    fig


def plot_capcurve(y_values, y_preds_proba):
    """
    This function plots CAP curve from model predictions and target
    Parameters:
    y_values: series with target values
    y_preds_proba: probabilities from binary classifier output
    """

    num_pos_obs = np.sum(y_values)
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x': [0, rate_pos_obs, 1], 'y': [0, 1, 1]})
    xx = np.arange(num_count) / float(num_count - 1)

    y_cap = np.c_[y_values, y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(drop=True)

    # print(y_cap_df_s.head(20))

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count - 1])  # add the first curve point (0,0) : for xx=0 we have yy=0

    percent = 0.5
    row_index = int(np.trunc(num_count * percent))

    val_y1 = yy[row_index]
    val_y2 = yy[row_index + 1]
    if val_y1 == val_y2:
        val = val_y1 * 1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index + 1]
        val = val_y1 + ((val_x2 - percent) / (val_x2 - val_x1)) * (val_y2 - val_y1)

    sigma_ideal = 1 * xx[num_pos_obs - 1] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
    sigma_model = integrate.simps(yy, xx)
    sigma_random = integrate.simps(xx, xx)

    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    # ar_label = 'ar value = %s' % ar_value
    # fig = plt.figure(figsize=(28,20))
    ax, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(35, 15))
    ax2.plot(ideal['x'], ideal['y'], color='grey', label='Perfect Model')
    ax2.plot(xx, yy, color='red', label='User Model')
    # ax.scatter(xx,yy, color='red')
    ax2.plot(xx, xx, color='blue', label='Random Model')
    ax2.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax2.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1,
             label=str(val * 100) + '% of positive obs at ' + str(percent * 100) + '%')

    '''
    fig.add_subplot(axes2)
    fig.xlim(0, 1.02)
    fig.ylim(0, 1.25)
  
    fig.title("CAP Curve - a_r value ="+str(ar_value))
    fig.xlabel('% of the data')
    fig.ylabel('% of positive obs')
    fig.legend()
    fig.show()
    '''

    def timeit(method):
        """
        This is a decorator that measures execution time of decorated function
        """

    def timeit(func):
        @wraps(func)
        def _time_it(*args, **kwargs):
            start = int(round(time.process_time() * 1000))
            try:
                return func(*args, **kwargs)
            finally:
                end_ = int(round(time.process_time() * 1000)) - start
                print(
                    f"Total execution time {func.__name__}: {end_ if end_ > 0 else 0} ms"
                )

        return _time_it
