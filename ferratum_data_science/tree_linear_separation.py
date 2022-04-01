# import pandas as pd
import numpy as np
import seaborn as sns
# from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt
# from numpy import sort
import statsmodels.formula.api as sm


def backward_elimination(x, y, sl, columns):

    """
    This function returns a dataframe and list of column names which P-value i lower than a then given threshhold
    Parameters:
    x: numpy array of values of features
    Y: numpy array of target values
    SL: threshold for P-value
    columns: list of column names
    """
    
    num_vars = len(x[0])
    for i in range(0, num_vars):
        regressor_ols = sm.OLS(y, x).fit()
        max_var = max(regressor_ols.pvalues).astype(float)
        if max_var > sl:
            for j in range(0, num_vars - i):
                if regressor_ols.pvalues[j].astype(float) == max_var:
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_ols.summary()
    return x, columns


def corr_features(cr, leave_uncorr=True, corr_threshold=0.9):

    """

    This function returns a  list of column names which has higher correlation than a then given threshold
    Parameters: cr: correlation matrix which is output of corr() function leave_uncorr: if leave_uncorr = True than
    output represents uncorrelated features, if leave_uncorr = False han output represents higher the threshold
    correlated features corr_threshold: minimum correlation value when we treat fetures as highly correlated

    """
    
    columns = np.full((cr.shape[0],), leave_uncorr, dtype=bool)
    for i in range(cr.shape[0]):
        for j in range(i+1, cr.shape[0]):
            if cr.iloc[i, j] >= corr_threshold:
                if leave_uncorr:
                    if columns[j]:
                        columns[j] = False
                else:
                    if not columns[j]:
                        columns[j] = True
    return columns

    
def scatterplot(timeline, df, column):

    """
    This function plots feature value distribution over time as scatterplot
    Parameters:
    timeline: name of column which represents date of observation
    df: dataframe with date and features columns
    column: name of column which we want to analyze
    """
    
    ts = df.set_index(timeline)
    tspart = ts.iloc[:, ts.columns.get_loc(column)]
    tspart.plot(figsize=(40, 40), style='k.')   # , style='k.'
    plt.show()


def target_distribution(rows, cols, data, target):

    """

    This function plots one chart for each column in given dataframe, which shows feature values distribution against
    target Parameters: rows: number of rows for arrangement of charts cols: number of columns for arrangement of
    charts data: dataframe with values of features for chart generation target: series with target values

    """
    
    fig = plt.figure(figsize=(20, 40))
    j = 0
    for i in data.columns:
        plt.subplot(rows, cols, j+1)
        j += 1
        sns.distplot(data[i][target == 0], color='r', label='bad')
        sns.distplot(data[i][target == 1], color='g', label='good')
        plt.legend(loc='best')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()


def variable_distribution(df, var):

    """
    This function plots distribution of values of given feature
    Parameters:
    df: dataframe with features
    var: name of the feature for plotting
    """
    
    plt.figure(figsize=(18, 14))
    plt.scatter(range(df.shape[0]), np.sort(df[var]))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title("Feature Distribution")
    plt.show()
