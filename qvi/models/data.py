import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

def make_lifeexpect(filepath):
    
    data = pd.read_csv(filepath).iloc[:,3:].dropna()
    data = data[data.columns[::-1]]
    D = data.shape[-1] - 1
    
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    (std_X, ustd_X), (std_Y, ustd_Y), _,_ = make_standardize_funs(X, y)
    data = pd.DataFrame(np.concatenate([std_X(X).values,np.expand_dims(std_Y(y).values,1)],1))
    return data, D

def make_forestfires(filepath):
    data = pd.read_csv(filepath, sep=',').iloc[:, 4:]
    D = data.shape[-1] - 1
    return data, D


def make_frisk(filepath, crime=2, precinct_type=1):
    """ crime: 1=violent, 2=weapons, 3=property, 4=drug
        eth  : 1=black, 2 = hispanic, 3=white
        precincts: 1-75
        precinct_type = (0, .1], (.1, .4], (.4, 1.]
    """
    df = pd.read_csv(filepath, skiprows=6, delim_whitespace=True)

    popdf = df[['pop', 'precinct', 'eth']]. \
        groupby(['precinct', 'eth'])['pop'].apply(sum)
    percent_black = np.array([popdf[i][1] / float(popdf[i].sum())
                              for i in np.arange(1, 76)])
    precinct_types = pd.cut(percent_black, [0, .1, .4, 1.])  #
    df['precinct_type'] = precinct_types.codes[df.precinct.values-1]
    dataframe = df[(df['crime'] == crime) & (
        df['precinct_type'] == precinct_type)]

    return dataframe

def make_boston():
    columns = load_boston()['feature_names']
    cols = ['CRIM', 'ZN', 'RM', 'AGE', 'DIS', 'RAD',
            'TAX', 'PTRATIO', 'LSTAT']
    X = pd.DataFrame(load_boston()['data'],columns = load_boston()['feature_names'])[cols].values
    y = load_boston()['target']
    data = pd.DataFrame(np.concatenate([X,np.expand_dims(y,1)],axis=1))
    D = X.shape[-1]
    return data, D

def make_standardize_funs(Xtrain, Ytrain):
    """ functions to scale/unscale data """
    # first create scale functions
    std_Xtrain = np.std(Xtrain, 0)
    std_Xtrain[std_Xtrain == 0] = 1.
    mean_Xtrain = np.mean(Xtrain, 0)

    std_Ytrain = np.std(Ytrain, 0)
    mean_Ytrain = np.mean(Ytrain, 0)

    def std_X(X): return (X - mean_Xtrain) / std_Xtrain
    def ustd_X(X): return X*std_Xtrain + mean_Xtrain

    def std_Y(Y): return (Y - mean_Ytrain) / std_Ytrain
    def ustd_Y(Y): return Y*std_Ytrain + mean_Ytrain
    return (std_X, ustd_X), (std_Y, ustd_Y), \
           (mean_Xtrain, std_Xtrain), (mean_Ytrain, std_Ytrain)
