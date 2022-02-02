'''
Script to test the function of churn_library

Author: Chaitanya Kanth Vadlapudi
Date: XX-XX-XXXX

'''

import os
import logging
import configparser
from ast import literal_eval
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    Input:
            import_data: reference to import_data function

    Output:
           data - dataframe
    '''
    try:
        data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return data


def test_eda(perform_eda, data):
    '''
    test perform eda function
    Input:
            perform_eda: reference to perform_eda function
            data - dataframe

    Output:
           None
    '''
    try:
        perform_eda(data)
        assert len(os.listdir('./images/eda/')) == 5
        logging.info(
            "Testing perform_eda: SUCCESS - EDA has generated 5 files")

    except AssertionError as err:
        logging.error(
            "Testing perform_eda: ERROR - EDA didnt generate 5 files")
        raise err


def test_encoder_helper(encoder_helper, data, cat_columns, response_var):
    '''
    test encoder helper
    Input:
            encoder_helper: reference to encoder_helper function
            data - dataframe
            cat_columns - categorical columns
            response_var - Target Variable

    Output:
           data - dataframe
    '''
    try:
        data = encoder_helper(data, cat_columns, response_var)
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info(
            "Testing encoder_helper - SUCCESS - Encoding of category columns is successful")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper - ERROR - Some issue in encoding the category columns")
        raise err

    try:
        updated_columns = list(data.columns)
        modified_cols = [col + "_" + response_var for col in cat_columns]

        assert set(modified_cols) - set(updated_columns) == set()
        logging.info(
            "Testing encoder_helper - SUCCESS - The anticipated number of category columns have been encoded")

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper - ERROR - The anticipated number of category columns have not been encoded")
        raise err

    return data


def test_perform_feature_engineering(
        perform_feature_engineering,
        data,
        response_var):
    '''
    test perform_feature_engineering
    Input:
            perform_feature_engineering: reference to perform_feature_engineering function
            data - dataframe
            response_var - Target Variable

    Output:
           x_train - Train dataset
           x_test - Test dataset
           y_train - Target train data
           y_test -  Target test data
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data, response_var)

        assert x_train.shape[0] + x_test.shape[0] == data.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == data.shape[0]
        assert x_train.shape[1] == x_test.shape[1]

        logging.info(
            "Testing perform_feature_engineering - SUCCESS - Train Test Split was successul")

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering - ERROR - Train Test Split was not successul")
        raise err

    return (x_train, x_test, y_train, y_test)


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    Input:
            train_models: reference to train_models function
            x_train - train dataframe
            x_test - test dataframe
            y_train - target train dataframe
            y_test - target test dataframe

    Output:
           None
    '''
    try:
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/rfc_model.pkl")
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./images/results/roc_curve_result.png")

        logging.info(
            "Testing train_models - SUCCESS - Models were trained successully")

    except AssertionError as err:
        logging.error("Testing train_models - ERROR - Models training failed")
        raise err


def test_load_saved_models(load_saved_models):
    '''
    test loading of saved models
    Input:
            load_saved_models: reference to load_saved_models function

    Output:
           rfc_model - Random forest model
           lr_model - Logistic Regression model

    '''
    try:
        rfc_model, lr_model = load_saved_models()
        assert rfc_model is not None
        assert lr_model is not None
        logging.info(
            "Testing load_saved_models - SUCCESS - Models were loaded successully")

    except AssertionError as err:
        logging.error(
            "Testing load_saved_models - ERROR - Models were not loaded successully")
        raise err

    return rfc_model, lr_model


def test_make_predictions(
        make_predictions,
        rfc_model,
        lr_model,
        x_train,
        x_test):
    '''
    test making predictions on train & test data
    Input:
            make_predictions: reference to make_predictions function
            rfc_model - Random forest model
            lr_model - Logistic Regression model
            x_train - train dataframe
            x_test - test dataframe

    Output:
           y_train_preds_rf - Train Predictions by RF model
           y_test_preds_rf - Test Predictions by RF model
           y_train_preds_lr - Train Predictions by LR model
           y_test_preds_lr - Test Predictions by LR model
    '''

    try:
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = make_predictions(
            rfc_model, lr_model, x_train, x_test)
        assert y_train_preds_rf.shape[0] == y_train_preds_lr.shape[0]
        assert y_test_preds_rf.shape[0] == y_test_preds_lr.shape[0]

        logging.info(
            "Testing make_predictions - SUCCESS - Predictions were made successully")

    except AssertionError as err:

        logging.error(
            "Testing make_predictions - ERROR - Predictions were not made successully")
        raise err

    return (
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr)


def test_classification_report_image(classification_report_image,
                                     y_train, y_test, y_train_preds_lr,
                                     y_train_preds_rf, y_test_preds_lr,
                                     y_test_preds_rf):
    '''
    test classification report of RF and Logistic regression models
    Input:
            classification_report_image: reference to classification_report_image function
            y_train_preds_rf - Train Predictions by RF model
            y_test_preds_rf - Test Predictions by RF model
            y_train_preds_lr - Train Predictions by LR model
            y_test_preds_lr - Test Predictions by LR model

    Output:
           None
    '''
    try:
        classification_report_image(y_train, y_test, y_train_preds_lr,
                                    y_train_preds_rf, y_test_preds_lr,
                                    y_test_preds_rf)

        assert os.path.isfile('./images/results/logistic_results.png')
        assert os.path.isfile('./images/results/rf_results.png')

        logging.info(
            "Testing classification_report_image - SUCCESS - Plots saved successfully")

    except AssertionError as err:
        logging.error(
            "Testing classification_report_image - ERROR - Plots were not saved successfully")
        raise err


def test_feature_importance_plot(
        feature_importance_plot,
        rfc_model,
        x_train,
        output_path):
    '''
    test feature importance plot of RF model
    Input:
            feature_importance_plot: reference to feature_importance_plot function
            rfc_model - Random forest model
            x_train - train dataframe
            output_path - path to save the feature importance plot

    Output:
           None
    '''
    try:
        feature_importance_plot(rfc_model, x_train, output_path)

        assert os.path.isfile(output_path)
        logging.info(
            "Testing feature_importance_plot - SUCCESS - Plots were saved successfully")

    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot - ERROR - Plots were not saved successfully")
        raise err


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./data/constants.ini')

    DATA = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA)

    CAT_COLUMNS = literal_eval(config['data']['CAT_COLUMNS'])
    QUANT_COLUMNS = literal_eval(config['data']['QUANT_COLUMNS'])
    RESPONSE_VAR = literal_eval(config['data']['RESPONSE_VAR'])

    print(type(CAT_COLUMNS))
    print(RESPONSE_VAR)
    DATA = test_encoder_helper(
        cls.encoder_helper,
        DATA,
        CAT_COLUMNS,
        RESPONSE_VAR)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA, RESPONSE_VAR)

    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    RFC_MODEL, LR_MODEL = test_load_saved_models(cls.load_saved_models)

    Y_TRAIN_PREDS_RF, Y_TEST_PREDS_RF, Y_TRAIN_PREDS_LR, Y_TEST_PREDS_LR = test_make_predictions(
        cls.make_predictions, RFC_MODEL, LR_MODEL, X_TRAIN, X_TEST)

    test_classification_report_image(
        cls.classification_report_image,
        Y_TRAIN,
        Y_TEST,
        Y_TRAIN_PREDS_LR,
        Y_TRAIN_PREDS_RF,
        Y_TEST_PREDS_LR,
        Y_TEST_PREDS_RF)

    test_feature_importance_plot(
        cls.feature_importance_plot,
        RFC_MODEL,
        X_TRAIN,
        "./images/results/feature_importances.png")
