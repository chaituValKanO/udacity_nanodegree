'''
Python Script to predict customer churn

Author: Chaitanya Kanth Vadlapudi
Date: 29-01-2022

'''
import os
from ast import literal_eval
import configparser

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

config = configparser.ConfigParser()
config.read('./data/constants.ini')


def import_data(filepath):
    '''
    returns dataframe for the csv found at filepath

    input:
            filepath: a path to the csv
    output:
            data: pandas dataframe
    '''
    data = pd.read_csv(filepath)
    return data


def perform_eda(data):
    '''
    perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Understanding Churn feature
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.savefig('./images/eda/churn_distribution.png')

    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.distplot(data['Total_Trans_Ct'])
    plt.savefig('./images/eda/total_transaction_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    '''
    response = response if response is not None else 'Churn'

    for cat_col_name in category_lst:
        new_column_name = cat_col_name + "_" + response
        temp_list = []

        group = data.groupby(cat_col_name).mean()[response]
        for val in data[cat_col_name]:
            temp_list.append(group.loc[val])

        data[new_column_name] = temp_list

    return data


def perform_feature_engineering(data, response):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    response = response if response is not None else 'Churn'

    keep_cols = literal_eval(config['data']['KEEP_COLUMNS'])
    x_data = pd.DataFrame()
    x_data[keep_cols] = data[keep_cols]
    y_target = data[response]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_target, test_size=0.3, random_state=42)
    return (x_train, x_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='liblinear')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # Saving the ROC curve
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')

    # Save the model as pickle files
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def load_saved_models():
    '''
    Load saved ML models

    input: None

    output:
            rfc_model - loaded Random Forest model
            lr_model -  loaded LOgistic regression model

    '''
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    return rfc_model, lr_model


def make_predictions(rfc_model, lr_model, x_train, x_test):
    '''
    Make predictions on train and test data using the saved models

    input:
          rfc_model - loaded Random Forest model
          lr_model -  loaded Logistic regression model
          x_train - Train data
          x_test  -  Test data

    output:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    '''

    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    return (
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr)
