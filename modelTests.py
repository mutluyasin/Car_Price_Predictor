import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from preProcessing import *
import joblib
from sklearn.model_selection import GridSearchCV


def model_selection(X, y):
    features = ["Brand", "Series", "Model", "Year", "Gear Type", "Kilometer", "Fuel Type", "Color", "Engine Volume",
                "Engine Power", "Body Type", "Drive", "Fuel Tank", "Paint-changed"]

    X = X[features]
    X_encoded = pd.get_dummies(X)

    param_grid = {
        'n_estimators': [100, 150, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'max_depth': [3, 5, 7, 11, 12],
        'subsample': [0.8, 0.7, 0.9, 1.0]
    }

    grid_search = GridSearchCV(estimator=XGBRegressor(), param_grid=param_grid, cv=3,
                               scoring='neg_mean_absolute_percentage_error')

    print('train started')
    grid_search.fit(X_encoded, y)
    best_model = grid_search.best_estimator_
    print('training finished')

    print('prediction')
    y_pred = best_model.predict(X_encoded)
    print('prediction finished')

    maep_test = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(grid_search.best_params_)
    print(r2)
    print(maep_test)
    print(average_error(y_pred, y))


def regression_test_with_one_hot_encoding(X, y, regressor, r=85):
    features = ["Brand", "Series", "Model", "Year", "Gear Type", "Kilometer", "Fuel Type", "Color", "Engine Volume",
                "Engine Power", "Body Type", "Drive", "Fuel Tank", "Paint-changed"]

    X = X[features]

    # one-hot-encoding
    X_encoded = pd.get_dummies(X)
    #print(list(X_encoded.keys()))


    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1, random_state=r)

    X_test = X_test.to_numpy().astype(int)
    y_test = y_test.to_numpy().astype(int)
    X_train = X_train.to_numpy().astype(int)
    y_train = y_train.to_numpy().astype(int)

    # Initialize the linear regression model
    model = regressor

    # Train the model
    print('train started')
    model.fit(X_train, y_train)
    print('train finished')

    # Make predictions on the test set
    #print('prediction started')
    y_pred_test = model.predict(X_test)
    #print('prediction finished')

    #print('calcualting performing criterias')

    # Evaluate the model

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    maep_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    avg_er_test = average_error(y_test, y_pred_test)

    #print("Mean Squared Error:", mse)
    #print("R^2 Score:", r2)  # accuracy

    print(X_test[10], y_test[10])
    print(model.predict([X_test[10]]))

    #print(X_test[100], y_test[100])
    #print(model.predict([X_test[100]]))

    #print(y_test[150])
    #print(model.predict([X_test[150]]))

    scores = [mse_test, mae_test, maep_test, r2_test,avg_er_test]

    print(scores)

    return model


def combine_models(X, y, models, regressor):
    features = ["Brand", "Series", "Model", "Year", "Gear Type", "Kilometer", "Fuel Type", "Color", "Engine Volume",
                "Engine Power", "Body Type", "Drive", "Fuel Tank", "Paint-changed"]

    X = X[features]
    X_encoded = pd.get_dummies(X)  #
    X_encoded = X_encoded.to_numpy()
    y = y.to_numpy()

    new_X = models[0].predict(X_encoded).reshape(-1, 1)

    for i in range(1, len(models)):
        prediction = models[i].predict(X_encoded).reshape(-1, 1)
        new_X = np.concatenate((new_X, prediction), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = regressor

    # Train the model
    #print('combine train started')
    model.fit(X_train, y_train)
    #print('train finished')

    # Make predictions on the test set
    #print('prediction started')
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    #print('prediction finished')

    #print('calcualting performing criterias')
    # Evaluate the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    maep_train = mean_absolute_percentage_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    maep_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    avg_er_train = average_error(y_train, y_pred_train)
    avg_er_test = average_error(y_test, y_pred_test)

    #print("Mean Squared Error:", mse)
    #print("R^2 Score:", r2)  # accuracy

    #print(X_test[10], y_test[10])
    #print(model.predict([X_test[10]]))

    #print(X_test[100], y_test[100])
    #print(model.predict([X_test[100]]))

    print(y_test[150])
    print(model.predict([X_test[150]]))
    print(regressor)

    # Optional: Print coefficients and intercept for linear regression
    #print("Coefficients:", model.coef_)
    #print("Intercept:", model.intercept_)

    print([mse_test, mae_test, maep_test, r2_test, mse_train, mae_train, maep_train, r2_train, avg_er_train,
           avg_er_test])

    return model


def average_error(y_real, y_pred):
    error_sum = 0
    N = y_real.shape[0]
    for i in range(N):
        error_sum += abs(y_real[i] - y_pred[i]) * 100 / y_real[i]

    return error_sum / N


def regressor_testing(X, y):
    regressors = [LinearRegression(), DecisionTreeRegressor(random_state=42),
                  RandomForestRegressor(n_estimators=100, random_state=42),
                  XGBRegressor(learning_rate=0.1, max_depth=11, n_estimators=300, subsample=1, n_jobs=-1),
                  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
                  AdaBoostRegressor(random_state=42, n_estimators=50), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
                  MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=3000,
                               random_state=42),
                  MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='adam', max_iter=3000,
                               random_state=42)
                  ]

    with open('regressionResults.txt', 'a') as f:

        for regressor in regressors:
            scores = regression_test_with_one_hot_encoding(X, y, regressor)
            score_names = [' mse_test: ', ' mae_test: ', ' maep_test: ', ' r2_test: ', ' mse_train: ', ' mae_train:'
                , ' maep_train:', ' r2_train:', ' avg_error_train ', ' avg_error_test ']

            text = 'regressor: ' + str(regressor)
            for i in range(len(score_names)):
                text += ' ' + score_names[i] + ' ' + str(scores[i])

            f.write(text + '\n')
            print(text + '\n')


def save_model(model):
    joblib.dump(model, 'car_price_prediction_model.pkl')




if __name__ == "__main__":

    #X, y = load_data()
    #regressor = XGBRegressor(learning_rate=0.1, max_depth=13, n_estimators=300, subsample=1, n_jobs=-1)

    #model = regression_test_with_one_hot_encoding(X, y, regressor)
    #save_model(model)
    # regressor_testing(X,y)
    # model_testing()
    # model_selection(X, y)
    pass