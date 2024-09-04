import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport


def csv_to_array(fileName):
    dataSet = pd.read_csv(fileName, dtype=str)  # load file data to a pandas array

    #  shuffles the data
    dataSet = dataSet.sample(frac=1, random_state=42).reset_index(drop=True)

    # split data to feature and target
    X = dataSet.drop(columns=['Price'])
    y = dataSet['Price']

    features = ["Brand", "Series", "Year", "Model", "Gear Type", "Kilometer", "Fuel Type", "Color", "Engine Volume",
                "Engine Power", "Body Type", "Drive", "Paint-changed", "Fuel Tank"]

    X = X[features]
    y = y.str.replace('.', '').astype(int)

    #  clean the data and change data type
    X['Year'] = X['Year'].astype(int)
    X['Kilometer'] = X['Kilometer'].str.replace(' km', '').str.replace('.', '').astype(int)

    # handle missing values
    X['Model'] = X['Model'].replace('-', np.nan)
    X['Model'] = X['Model'].fillna(0)

    X['Engine Volume'] = X['Engine Volume'].replace('-', np.nan)
    X['Engine Volume'] = X['Engine Volume'].fillna(0)

    #  define a func to clean the engine volume feature and change data type
    def clean_engine_volume(volume):
        volume = str(volume)
        if ' - ' in volume:
            # Range values like '1201 - 1400 cm3'
            return int(volume.split('-')[1].strip().replace(' cm3', '').replace(' cc', '').replace('.', ''))
        elif '\' e kadar' in volume:
            # Values like '1200'e kadar'
            return int(volume.split('\'')[0].strip().replace(' cm3', '').replace(' cc', '').replace('.', ''))
        elif 've üzeri' in volume:
            # Values like '6001 ve üzeri'
            return int(volume.split(' ')[0].strip().replace(' cm3', '').replace(' cc', '').replace('.', ''))
        else:
            # Single value like '2000 cc'
            return int(volume.replace(' cm3', '').replace(' cc', '').replace('.', ''))

    X['Engine Volume'] = X['Engine Volume'].apply(clean_engine_volume)

    # handle missing values
    X['Engine Power'] = X['Engine Power'].replace('-', np.nan)
    X['Engine Power'] = X['Engine Power'].fillna(0)

    #  define a func to clean the engine power feature and change data type
    def clean_engine_power(power):
        # Convert to lower case to handle 'HP' and 'hp'
        power = str(power)
        power = power.lower().replace(' hp', '')

        if ' - ' in power:
            # Range values like '76 - 100 hp'
            parts = power.split(' - ')
            # Compute the average of the range
            return int((int(parts[0]) + int(parts[1])) / 2)
        elif '\'ye kadar' in power:
            # Values like '50\'ye kadar' (up to 50)
            return int(power.split('\'')[0].strip())
        elif 've üzeri' in power:
            # Values like '601 ve üzeri' (601 and above)
            return int(power.split(' ')[0].strip())
        else:
            # Single value like '56 hp'
            return int(power)

    X['Engine Power'] = X['Engine Power'].apply(clean_engine_power)

    # handle missing values
    X['Fuel Tank'] = X['Fuel Tank'].fillna(0)
    X['Fuel Tank'] = X['Fuel Tank'].astype(str).str.replace(' lt', '').astype(int)

    return X, y


def graph_price_vs_kilometer(X, y, series):
    # seperate the data that contains desired series
    X_series = X[X['Series'] == series]
    y_series = y[X['Series'] == series]

    #  plotting the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_series['Kilometer'], y_series, alpha=0.5)
    plt.title('Relationship between Price and Kilometer')
    plt.ylabel('Price')
    plt.xlabel('Kilometer')
    plt.grid(True)
    plt.show()


def graph_price_vs_year(X, y, series):
    # seperate the data that contains desired series
    X_series = X[X['Series'] == series]
    y_series = y[X['Series'] == series]

    #  plotting the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_series['Year'], y_series, alpha=0.5)
    plt.title('Relationship between Price and Year')
    plt.ylabel('Price')
    plt.xlabel('Year')
    plt.grid(True)
    plt.show()


def graph_price_vs_enginePower(X, y, series):
    # seperate the data that contains desired series
    X_series = X[X['Series'] == series]
    y_series = y[X['Series'] == series]

    #  plotting the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_series['Engine Power'], y_series, alpha=0.5)
    plt.title('Relationship between Price and Engine Power')
    plt.ylabel('Price')
    plt.xlabel('Engine Power')
    plt.grid(True)
    plt.show()


def ydata(X):
    # Generate the profiling report
    profile = ProfileReport(X, title='Profiling Report')

    # Save the report as an HTML file
    profile.to_file("profiling_report.html")


def binary_paintChanged(X):
    def helper(value):
        if 'Belirtilmemiş' in str(value) or 'orjinal' in str(value):
            return 0
        return 1

    X['Paint-changed'] = X['Paint-changed'].apply(helper)

    return X


def replace_zeros(X):
    ft_mean = np.mean(X['Fuel Tank'])
    X['Fuel Tank'] = X['Fuel Tank'].replace(0, ft_mean)

    ep_mean = np.mean(X['Engine Power'])
    X['Engine Power'] = X['Engine Power'].replace(0, ep_mean)

    ev_mean = np.mean(X['Engine Volume'])
    X['Engine Volume'] = X['Engine Volume'].replace(0, ev_mean)

    return X


def mean_for_Series(X, y):
    for series in X['Series'].unique():
        indices = X['Series'] == series
        X.loc[X['Series'] == series, 'Series'] = int(np.mean(y[indices]))

    return X


def mean_for_Model(X, y):
    for model in X['Model'].unique():
        indices = X['Model'] == model
        X.loc[X['Model'] == model, 'Model'] = int(np.mean(y[indices]))

    return X


def save_to_csv(X, y):
    data = pd.concat([X, y], axis=1)
    data.to_csv('processedData.csv', index=False)


def load_data():
    dataSet = pd.read_csv('processedData.csv')

    X = dataSet.drop(columns=['Price'])
    y = dataSet['Price']

    return X, y

def car_info_csv(X, y):
    data = pd.concat([X['Brand'], X['Series'], X['Model'], y],axis=1)
    data.to_csv('carInfo.csv',index=False)

if __name__ == '__main__':

    #X, y = csv_to_array('dataSet(original).csv')
    # print('reading is done')
    #X = binary_paintChanged(X)
    #X = replace_zeros(X)
    #car_info_csv(X,y)
    #X = mean_for_Series(X, y)
    #X = mean_for_Model(X, y)
    #save_to_csv(X,y)
    pass

