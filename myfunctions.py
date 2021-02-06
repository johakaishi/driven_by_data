# import packages
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


# # original print_scores function
# def print_scores(model, X_train, X_test, y_train, y_test):
#     '''print scores for train, test, CV mean for a fitted model'''
    
#     print('Train score: ', model.score(X_train, y_train))
#     print('Test score:  ', model.score(X_test, y_test))
#     print('CV mean:     ', cross_val_score(model, X_train, y_train, cv=5).mean())


# print_scores function with RMSE, MAE
def print_scores(model, X_train, X_test, y_train, y_test):
    '''print scores for train, test, CV mean for a fitted model'''

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print(f'R2 Train, Test:\t\t{model.score(X_train, y_train)} \t {model.score(X_test, y_test)}')
    print(f'R2 Train (CV Mean):\t{cross_val_score(model, X_train, y_train, cv=5).mean()}')
    print()
    print(f'RMSE Train, Test:\t{(mean_squared_error(y_train, pred_train))**0.5} \t {(mean_squared_error(y_test, pred_test))**0.5}')
    print(f'MAE  Train, Test:\t{mean_absolute_error(y_train, pred_train)} \t {mean_absolute_error(y_test, pred_test)}')
    

def plot_predictions(model, X_train, X_test, y_train, y_test):
    '''plot actual against predictions'''
    # model predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # max limits for plots
    max_train = [y_train.max() if y_train.max() > pred_train.max() else pred_train.max()][0]
    max_test = [y_test.max() if y_test.max() > pred_test.max() else pred_test.max()][0]

    # predictions against actual plots
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    sns.regplot(x=pred_train, y=y_train, color='blue', ax=ax[0])
    sns.regplot(x=pred_test, y=y_test, color='blue', ax=ax[1])
    
    ax[0].plot([0,max_train], [0,max_train], color='red')
    ax[1].plot([0,max_test], [0,max_test], color='red')

    ax[0].set_title('Actual / Predicted (Train)')
    ax[0].set_xlabel('Predicted Price (GBP)')
    ax[0].set_ylabel('Actual Price (GBP)')    
    ax[1].set_title('Actual / Predicted (Test)')
    ax[1].set_xlabel('Predicted Price (GBP)')
    ax[1].set_ylabel('Actual Price (GBP)')
    
    print('Predictions:')
    plt.show()
    
    
def plot_residuals(model, X_train, X_test, y_train, y_test):
    '''plot residuals: histogram, boxplot, and scatterplot of standardised residuals against predicted price'''

    # model predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # residuals
    res_train = y_train - pred_train
    res_test = y_test - pred_test

    # standardized residuals
    res_train_std = (res_train - res_train.mean()) / res_train.std()
    res_test_std = (res_test - res_test.mean()) / res_test.std()    

    # plot residuals
    fig, ax = plt.subplots(3,2, figsize=(16,14))
    plt.tight_layout(pad=5)

    # train
    sns.distplot(res_train, kde=False, color='blue', ax=ax[0,0])
    sns.boxplot(res_train, color='blue', ax=ax[1,0])
    sns.scatterplot(x=pred_train, y=res_train_std, color='blue', ax=ax[2,0])

    # test
    sns.distplot(res_test, kde=False, color='blue', ax=ax[0,1])
    sns.boxplot(res_test, color='blue', ax=ax[1,1])
    sns.scatterplot(x=pred_test, y=res_test_std, color='blue', ax=ax[2,1])

    # labels
    ax[0,0].set_title('Residuals (Train)')
    ax[0,0].set_ylabel('Occurences')    
    ax[2,0].set_title('Standardised Residuals (Train)')  
    ax[2,0].set_xlabel('Predicted Price (GBP)')  
    ax[2,0].set_ylabel("Standardised Residual (st dev's)")  
        
    ax[0,1].set_title('Residuals (Test)')
    ax[0,1].set_ylabel('Occurences')    
    ax[2,1].set_title('Standardised Residuals (Test)')
    ax[2,1].set_xlabel('Predicted Price (GBP)')  
    ax[2,1].set_ylabel("Standardised Residual (st dev's)")  

    print('Distribution of residuals:')
    plt.show()
    
    
def plot_coefficients(model, X_train, X_test, y_train, y_test):
    '''plot coefficients from a model'''
    # df for model coefficients
    df_model_coef = pd.DataFrame(model.coef_, index=X_train.columns, columns=['coef'])
    df_model_coef['coef_abs'] = np.abs(df_model_coef)

    # plot coefficients in order of importance
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    plt.tight_layout(w_pad=10)

    df_model_coef.coef.sort_values()[:10].plot(kind='barh', ax=ax[0])
    df_model_coef.coef.sort_values()[-10:].plot(kind='barh', ax=ax[1])

    ax[0].set_title('Feature Importance (negative impact)')
    ax[1].set_title('Feature Importance (positive impact)')

    print('Coefficients:')
    plt.show()

    
def plot_coef_pipe(pipe, X_train, X_test, y_train, y_test, features_cont):
    '''plot coefficients from a model'''
    # df for model coefficients    
    col_names = features_cont + list(transformer.named_transformers_.cat.get_feature_names())

    df_model_coef = pd.DataFrame(pipe.named_steps.model.coef_, index=col_names, columns=['coef'])
    df_model_coef['coef_abs'] = np.abs(df_model_coef)
    
    # plot coefficients in order of importance
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    plt.tight_layout(w_pad=10)

    df_model_coef.coef.sort_values()[:10].plot(kind='barh', ax=ax[0])
    df_model_coef.coef.sort_values()[-10:].plot(kind='barh', ax=ax[1])

    ax[0].set_title('Feature Importance (negative impact)')
    ax[1].set_title('Feature Importance (positive impact)')

    print('Coefficients:')
    plt.show()

    
def plot_all(model, X_train, X_test, y_train, y_test):
    '''plot predictions, residuals and coefficients'''
    plot_predictions(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    plot_residuals(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    plot_coefficients(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    
    