# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:40:10 2025

@author: Georg
"""
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
#from sklearn.model_selection import train_test_split
import numpy as np
from sympy import symbols, diff, lambdify
from sympy.solvers import solveset, S
import sys


## TODO: Fix the join on this function. The new_index needs to be from the other table.
def JoinTables(dfs: list, ons: list, new_index=False):

    counter = 0
    lst_outdfs = []
    for tuple_dfs in dfs:
        left_df = tuple_dfs[0]
        right_df = tuple_dfs[1]
        if(new_index):
            right_df = right_df.set_index(ons[counter], inplace=False)
            
        left_df = left_df.join(right_df, on=ons[counter], validate='m:1')
        lst_outdfs.append(left_df)
        counter += 1
    return lst_outdfs

def GetMedian(data: list) -> float: 
    """
    Gets the mean from a series. If there is not a center, it takes the 
    average of the higher and lower values.
    
    Parameters: 
        data (list): List of data
        
    Returns: 
        float: Median value of data

    """
    return stat.median(data)

def GetQuadrants(lst_data):
    q2 = GetMedian(lst_data)
    upper_part = [item for item in lst_data if item < q2]
    q1 = GetMedian(upper_part)

    lower_part = [item for item in lst_data if item > q2]
    q3 = GetMedian(lower_part)
    return(q1, q2, q3)
    
def PlotLine(x, y, title, xaxis_lbl, yaxis_lbl, filename):
    plt.figure()
    sns.lineplot(x=x, y=y)
    
    plt.xlabel(xaxis_lbl)
    plt.xticks(rotation=90)
    plt.ylabel(yaxis_lbl)
    plt.title(title)
    plt.savefig(filename)
    return None

def NormalizeValues(values, max_value, min_value):
    """
    Normalizes values inputed to be between 0 and 1. 

    Parameters
    ----------
    values : NUMPY ARRAY
        The values that need to be normalized.
    max_value : FLOAT OR INT
        Max value for dataset.
    min_value : FLOAT OR INT
        MIN value for dataset.

    Returns
    -------
    new : NUMPY ARRAY
        The normalized values with smallest 0 and largest 1.

    """
    new = (values - min_value)/(max_value - min_value)
    return new
    
def UnNormalizeValues(value, max_value, min_value):
    new = (max_value - min_value)*value + min_value
    return new
"""
def CalculateDerivative(coefs):
    deg = len(coefs)
    x = symbols('x')
    f = 0
    for i in range(0, deg):
        f = f + coefs[i]*pow(x, i+1)
    der = diff(f, x)
    
"""   
def KeepScore(f, aic_score, min_bound, max_bound):
    if(aic_score is None):
        return False
    x = symbols('x')
    lower = f.subs(x, min_bound)
    upper = f.subs(x, max_bound)
    if(lower < 0 or upper < 0 or lower > 1 or upper > 1):
        return False
    
    der = diff(f, x)
    res = solveset(der, x, domain=S.Reals)
    for value in res:
        if(min_bound <= value <= max_bound):
            y = f.subs(x, value)
            if((y < 0) or (y > 1)):
                return False
    
    return True

def CreateModelPoly(coefs, y_intercept):
    """
    Creates the function (with x symbol) of the model using 
    ceofficients and y intercept given from sklearn Regression

    Parameters
    ----------
    coefs : list
        list of a_1, a_2, a_3, ... params.
    y_intercept : number
        the y intercept for the model.

    Returns
    -------
    y : expression of equation with x as the variable
        Example: If passed coefs = [a_1, a_2], y_intercept = a_0
        Returns a_0 + a_1*x + a_2*x**2 
    """
    x = symbols('x')
    deg = len(coefs)
    y = y_intercept
    for i in range(0, deg):
        y = y + coefs[i]*pow(x, i+1)
    return y

def CalculateAICBIC(y_actual, y_pred, k, criteria='AIC'):
    mse = mean_squared_error(y_actual, y_pred)
    n = len(y_actual)
    if(mse == 0):
        return None
    
    log_likelihood = (-n / 2 * np.log(2 * np.pi * mse) - 
                          (1 / (2 * mse)) *np.sum((y_actual - y_pred) ** 2))
    
    if(criteria.capitalize() == 'BIC'):
        return np.log(n) * k - 2 * log_likelihood
    else:
        return 2 * k - 2 * log_likelihood
    
def AddNoise(X):
    for i in range(0, len(X)):
        error = np.random.normal()
        X[i] = X[i] + error
    return X

def RunRegressionMethods(X, y, degrees=[1, 2], error_preface=""):
    """
    Run regression methods for a variety of degrees. Calculate the 
    AIC and BIC scores. Determine which degree method is best.
    This method uses the SKlearn regression methods and AIC calculated 
    from ***** FILL IN HERE THE LIKELIHOOD CALCULATION *****

    Parameters
    ----------
    X : array
        Independent variables.
    y : array
        dependent variables.
    degrees : list, optional
        Degrees to run regression on. The default is [1, 2, 3].

    Returns
    -------
    Expression a_0 + a_1x + a_2x^2 +....

    """
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    current_score = 1000000000
    winning_f = None
    for degree in degrees:

        (reg_model, y_pred) = TrainModel(X, y, degree)
        stdout_old = sys.stdout
        log_file = open("log_file", "a")
        sys.stdout = log_file
        
        # Determine if the model should be thrown out or not.
        aic = CalculateAICBIC(y, y_pred, degree)
        if(aic is None):
            print(error_preface, " MSE was 0 for degree: ", degree)
        params = list(reg_model.coef_.reshape(-1))
        params.pop(0)
        y_intercept = reg_model.intercept_[0]
        f = CreateModelPoly(params, y_intercept)
        bKeepScore = KeepScore(f, aic, 0, 1)
        if((bKeepScore is False) and (aic is not None)):
            print(error_preface, "Model failed because out of bounds for degree ", degree)
            
        if(bKeepScore and (aic < current_score)):
            # This is the lowest AIC score and the model is within bounds.
            current_score = aic
            winning_f = f
        
        log_file.close()
        sys.stdout = stdout_old
    return winning_f

def TrainModel(input_known, output_known, degree):
    model = LinearRegression()
    poly = PolynomialFeatures(degree=degree)

    X_transformed = poly.fit_transform(input_known)
    model.fit(X_transformed, output_known)
    y_pred = model.predict(X_transformed)
    
    return (model, y_pred)

def FindMissingValue(X_missing, f):
    x = symbols('x')
    y = lambdify(x, f, modules=["scipy", "numpy"])
    res = y(X_missing)
    return res




        
