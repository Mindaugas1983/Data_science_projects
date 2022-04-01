import pandas as pd
import numpy as np
# from pandas import Series
from matplotlib import pyplot as plt


def autolabel(rects, ax, color='red'):
    """
    This function is additional function of "plot_distribution". It helps to proper label x axis
    Parameters:
    rects: array of values to apply on subplot
    ax: pyplot sublot on which we need to apply changes
    color: color of bar
    """

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom', bbox=dict(facecolor=color, alpha=0.5))


def plot_distribution(df, x=0):
    """
    This function makes bar plot of categorical variable for each category correlation with target Parameters: df:
    Dataframe where first column is a categorical variable for ploting and second column is a target x: type of
    categorical variable, if x = 0 categorical variable is expected to be numeric , otherwise it can be string
    """

    if x == 0:
        df_good = df[df.iloc[:, 1] == 1]
        df_bad = df[df.iloc[:, 1] == 0]
        x_axis_g = np.sort(df_good.iloc[:, 0].unique())  # .sort_values()
        y_axis_g = np.bincount(df_good.iloc[:, 0])
        x_axis_b = np.sort(df_bad.iloc[:, 0].unique())  # .sort_values()
        y_axis_b = np.bincount(df_bad.iloc[:, 0])
        x_axis = np.sort(df.iloc[:, 0].unique())  # .sort_values()
        y_axis = np.bincount(df.iloc[:, 0])
        fig, ax = plt.subplots(figsize=(40, 20))
        bar_width = 0.25
        opacity = 0.4
        rects1 = ax.bar(x_axis_g, y_axis_g, bar_width, alpha=opacity, color='g', label='Good')
        rects2 = ax.bar(x_axis_b + 0.3, y_axis_b, bar_width, alpha=opacity, color='r', label='Bad')
        rects3 = ax.bar(x_axis - 0.3, y_axis, bar_width, alpha=opacity, color='y', label='All')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Amount')
        ax.set_xticks(x_axis_g + bar_width / 2)
        ax.set_xticklabels(x_axis_g)
        autolabel(rects1, ax, 'g')
        autolabel(rects2, ax, 'r')
        autolabel(rects3, ax, 'y')
        ax.legend()
        fig.tight_layout()
        plt.show()
    else:
        df_good = df[df.iloc[:, 1] == 1]
        df_bad = df[df.iloc[:, 1] == 0]
        x_axis_g, y_axis_g = np.unique(df_good.iloc[:, 0].astype(str), return_counts=True)
        x_axis_b, y_axis_b = np.unique(df_bad.iloc[:, 0].astype(str), return_counts=True)
        x_axis, y_axis = np.unique(df.iloc[:, 0].astype(str), return_counts=True)
        x_g, x_g_name = pd.factorize(x_axis_g)
        x_b, x_b_name = pd.factorize(x_axis_b)
        x_a, x_a_name = pd.factorize(x_axis)
        fig, ax = plt.subplots(figsize=(40, 20))
        bar_width = 0.25
        opacity = 0.4
        rects1 = ax.bar(x_g, y_axis_g, bar_width, alpha=opacity, color='g', label='Good')
        rects2 = ax.bar(x_b + 0.3, y_axis_b, bar_width, alpha=opacity, color='r', label='Bad')
        rects3 = ax.bar(x_a - 0.3, y_axis, bar_width, alpha=opacity, color='y', label='All')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Amount')
        ax.set_xticks(ticks=x_g)
        ax.set_xticklabels(x_axis_g)
        autolabel(rects1, ax, 'g')
        autolabel(rects2, ax, 'r')
        autolabel(rects3, ax, 'y')
        ax.legend()
        fig.tight_layout()
        plt.show()


def calculate_other(row):
    """
    This function is additional function for "one_hot"and "one_hot_corr" which produces column named "rest" which
    means that all dummy variables for this category and for this observation was equal 0 Parameters: row: a row for
    checking if all columns are equal to 0
    """

    if row.sum() == 0:
        return 1
    return 0


def one_hot(df, column, threshold, prefix, column_other, null_value):
    """
    This function converts categorical variable to one hot encoding and removes original column Parameters: df:
    dataframe with categorical column which needs to be converted to one-hot encoding column: name of the column
    which needs to be converted  to one-hot encoding threshold: minimum amount of observations needed for a
    particular value in order to make dummy variable from that value prefix: prefix for newly created columns
    column_other: if column_other = 1 then dummy variable will be created which represents that this observation
    belongs or not belongs to any other dummy variable, otherwise such column will not be created null_value: value
    in which missing values will be replaced
    """

    # df_temp = pd.DataFrame()
    df_temp = df.copy(deep=True)
    df_temp[column] = df_temp[column].fillna(null_value)
    counts = df_temp[column].value_counts()
    ep_mask = df_temp[column].isin(counts[counts > threshold].index)
    ep_dummies = pd.get_dummies(df_temp[column][ep_mask])
    length = ep_dummies.shape[1]
    df_temp = df_temp.join(ep_dummies, lsuffix='_df2', rsuffix='_ep_dummies')
    ep_dummies2 = df_temp.iloc[:, df_temp.shape[1] - length:]
    ep_dummies2 = ep_dummies2.fillna(0)
    if column_other == 1:
        ep_dummies2['rest'] = ep_dummies2.apply(lambda row: calculate_other(row), axis=1)
    ep_dummies2 = ep_dummies2.add_prefix(prefix)
    df_final = df_temp.iloc[:, : df_temp.shape[1] - length].join(ep_dummies2)
    del df_final[column]
    return df_final


def corr_with_target(feature, target):
    """
    This function returns data about each value of categorical variable correlation with target. Column "sum"
    represents count of observations with this value, column "corr"represents a fraction of good customers in full
    sample that have this value Parameters: feature: series with values of categorical variable which need to be
    analyzed target: series with target values in the same order as "feature"
    """

    check = pd.crosstab(feature, target)
    check["sum"] = check[0] + check[1]
    check["corr"] = check[1] / check["sum"]
    return check


def corr_features(corr_list, min_amount, b_min, b_max):
    """
    This function produces dataframe with these values of categorical feature which meets given thresholds
    Parameters: corr_list: correlation list produced by function "corr_with_target" min_amount: threshold for minimum
    amount of observations when value of categorical variable should be considered for one hot encoding b_min:
    threshold value for correlation - if correlation is lower than this value then value of categorical variable
    should be considered for one hot encoding b_max: threshold value for correlation - if correlation is higher than
    this value then value of categorical variable should be considered for one hot encoding
    """

    for_variables = corr_list[
        (corr_list['sum'] > min_amount) & ((corr_list['corr'] > b_max) | (corr_list['corr'] < b_min))]
    return for_variables


def one_hot_corr(source_df, corr_df, coll_name, prefix, column_other, null_value):
    """
    This function converts categorical variable only for these values which are returned by function "corr_features"
    to one hot encoding and removes original column Parameters: source_df: dataframe with categorical column which
    needs to be converted to one-hot encoding corr_df: results from function "corr_features" coll_name: name of the
    column which needs to be converted  to one-hot encoding prefix: prefix for newly created columns column_other: if
    column_other = 1 then dummy variable will be created which represents that this observation belongs or not
    belongs to any other dummy variable, otherwise such column will not be created null_value: value in which missing
    values will be replaced
    """

    cr_mask = source_df[coll_name].isin(corr_df.index)
    cr_dummies = pd.get_dummies(source_df[coll_name][cr_mask])
    cr_dummies = cr_dummies.add_prefix(prefix)
    length = cr_dummies.shape[1]
    df_dm = source_df.join(cr_dummies)
    df_dm_part = df_dm.iloc[:, df_dm.shape[1] - length:]
    df_dm_part = df_dm_part.fillna(null_value)
    if column_other == 1:
        df_dm_part[prefix + '_rest'] = df_dm_part.apply(lambda row: calculate_other(row), axis=1)
    df_dm2 = df_dm.iloc[:, : df_dm.shape[1] - length].join(df_dm_part)
    del df_dm2[coll_name]
    return df_dm2


def compare_columns(required, existing):
    """
    This function prints column names which exist in first dataframe before "Target"  column, but do not exist in
    second dataframe  before "Target"  column Parameters: required: dataframe with columns we want to have that
    appears before "Target" column existing: dataframe with columns we want to check that appears before "Target" column
    """

    l1 = list(required.iloc[:, : required.columns.get_loc("Target")])
    l2 = list(existing.iloc[:, : existing.columns.get_loc("Target")])
    diff = list(set(l1) - set(l2))
    print(diff)


def required_columns(required1, required2):
    """
    This function prints distinct union of column names which exist in  first or in second dataframe before "Target"
    columns Parameters: required1: first dataframe with columns we want to have that appears before "Target" column
    required2: second dataframe with columns we want to have that appears before "Target" column
    """

    l1 = list(required1.iloc[:, : required1.columns.get_loc("Target")])
    l2 = list(required2.iloc[:, : required2.columns.get_loc("Target")])
    cols = list(set(l1) | set(l2))
    print(cols)


def get_duplicate_columns(df):
    """
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    Parameters:
    df = Dataframe object
    return = List of columns whose contents are duplicates.
    """
    duplicate_column_names = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            other_col = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])
    return list(duplicate_column_names)
