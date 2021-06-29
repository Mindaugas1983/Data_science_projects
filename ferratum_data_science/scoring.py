import os.path 
import dataiku
import pickle
import pandas as pd


def ld_model(folder, model_name):
    
    '''
    This function loads and returns model from dataiku directory
    Parameters:
    folder: path to folder where model is located
    model_name: name of model which we want to laoad
    '''
    
    handle = folder
    folder_path = handle.get_path()
    file_path = os.path.join(folder_path, model_name )
    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


def scale_df_mm(df, std_df):
    
    '''
    This function scales given dataframe using min max scaling based on given scaling table
    Parameters:
    df: dataframe for scaling
    std_df: scaling table which we want to use
    '''
    
    df2 = pd.DataFrame()
    for column in df.columns[:]:
        df2[column] = (df[column] - std_df.iloc[std_df.index.get_loc("min"), std_df.columns.get_loc(column)] ) / (std_df.iloc[std_df.index.get_loc("max"), std_df.columns.get_loc(column)] - std_df.iloc[std_df.index.get_loc("min"), std_df.columns.get_loc(column)])
    return df2


def scale_df_mean(df, std_df):
    
    '''
    This function scales given dataframe using standard scaling based on given scaling table
    Parameters:
    df: dataframe for scaling
    std_df: scaling table which we want to use
    '''
        
    df2 = pd.DataFrame()
    for column in df.columns[:]:
        df2[column] = (df[column] - std_df.iloc[std_df.index.get_loc("mean"), std_df.columns.get_loc(column)] ) / std_df.iloc[std_df.index.get_loc("std"), std_df.columns.get_loc(column)]
    return df2


def load_st(fld, name):
    
    '''
    This function loads and returns file from dataiku directory
    Parameters:
    fld: path to folder where file is located
    name: name of file which we want to laoad
    '''
    
    handle = dataiku.Folder(fld)
    folder_path = handle.get_path()
    file_path = os.path.join(folder_path, name)
    with open(file_path, 'rb') as f:
        loaded_std_df = pickle.load(f)
    return loaded_std_df

def save_keras_model(model, directory, name):
    
    '''
    This function saves given keras model to given folder. It saves model architecture and weighs in separate files
    Parameters:
    model: keras based model which needs to be saved
    directory: path to folder where we want to save model
    name: name of json and weights files 
    '''
    
    model_json = model.to_json()
    with open(directory + '\\' + name + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(directory + '\\' + name + '.h5')    
    print('model saved')


def load_keras_model(model, directory):
    
    '''
    This function loads keras model saved into two separate files
    Parameters:
    model: name of keras based model which needs to be loaded
    directory: path to folder from which we want to load model
    '''
    
    json_file = open(directory + '\\' + model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(directory + '\\' + model + '.h5')
    print('model_loaded')
    return loaded_model