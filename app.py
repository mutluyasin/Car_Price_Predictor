from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

def car_models():
    data = pd.read_csv('carInfo.csv')
    X = data.drop(columns=['Price'])
    y = data['Price'].astype(int)
    car = {}

    for brand in X['Brand'].unique():
        car[brand] = {}
        for series in X.loc[X['Brand'] == brand, 'Series'].unique():
            car[brand][series] = []
            for model in X.loc[(X['Brand'] == brand) & (X['Series'] == series), 'Model'].unique():
                if str(model) != 'nan':
                    car[brand][series].append(model)

    return car, X, y

def correct_data(user_data, X, y):

    keys = ['Series', 'Model', 'Year', 'Kilometer', 'Engine Volume', 'Engine Power', 'Fuel Tank', 'Paint-changed',
            'Brand_Alfa Romeo', 'Brand_Anadol', 'Brand_Askam', 'Brand_Aston Martin', 'Brand_Audi', 'Brand_BMC',
            'Brand_BMW', 'Brand_Bentley', 'Brand_Buick', 'Brand_Cadillac', 'Brand_Chery', 'Brand_Chevrolet',
            'Brand_Chrysler', 'Brand_Citroen', 'Brand_Cupra', 'Brand_DFM', 'Brand_DS Automobiles', 'Brand_Dacia',
            'Brand_Daewoo', 'Brand_Daihatsu', 'Brand_Dodge', 'Brand_Ferrari', 'Brand_Fiat', 'Brand_Ford',
            'Brand_GAZ', 'Brand_GMC', 'Brand_Geely', 'Brand_HF Kanuni', 'Brand_Honda', 'Brand_Hongqi', 'Brand_Hummer',
            'Brand_Hyundai', 'Brand_Ikco', 'Brand_Infiniti', 'Brand_Isuzu', 'Brand_Iveco - Otoyol', 'Brand_Jaguar',
            'Brand_Jeep', 'Brand_Joyce', 'Brand_Kia', 'Brand_Lada', 'Brand_Lamborghini', 'Brand_Lancia',
            'Brand_Land Rover', 'Brand_Lexus', 'Brand_Lincoln', 'Brand_MG', 'Brand_MINI', 'Brand_Mahindra',
            'Brand_Maserati', 'Brand_Mazda', 'Brand_Mercedes - Benz', 'Brand_Mitsubishi', 'Brand_Moskvitch',
            'Brand_Nissan', 'Brand_Opel', 'Brand_Peugeot', 'Brand_Pontiac', 'Brand_Porsche', 'Brand_Proton',
            'Brand_RKS', 'Brand_Regal Raptor', 'Brand_Renault', 'Brand_Rolls-Royce', 'Brand_Rover', 'Brand_SWM',
            'Brand_Saab', 'Brand_Seat', 'Brand_Seres', 'Brand_Skoda', 'Brand_Skywell', 'Brand_Smart', 'Brand_Ssangyong',
            'Brand_Subaru', 'Brand_Suzuki', 'Brand_TOGG', 'Brand_Tata', 'Brand_Temsa', 'Brand_Tesla', 'Brand_Tofaş',
            'Brand_Toyota', 'Brand_Volkswagen', 'Brand_Volta', 'Brand_Volvo', 'Gear Type_Düz', 'Gear Type_Otomatik',
            'Gear Type_Yarı Otomatik', 'Fuel Type_Benzin', 'Fuel Type_Dizel', 'Fuel Type_Elektrik', 'Fuel Type_Hibrit',
            'Fuel Type_LPG & Benzin', 'Color_-', 'Color_Altın', 'Color_Bej', 'Color_Beyaz', 'Color_Bordo',
            'Color_Diğer', 'Color_Füme', 'Color_Gri', 'Color_Gri (Gümüş)', 'Color_Gri (metalik)', 'Color_Gri (titanyum)',
            'Color_Kahverengi', 'Color_Kırmızı', 'Color_Lacivert', 'Color_Mavi', 'Color_Mavi (metalik)',
            'Color_Mor', 'Color_Pembe', 'Color_Sarı', 'Color_Siyah', 'Color_Turkuaz', 'Color_Turuncu', 'Color_Yeşil',
            'Color_Yeşil (metalik)', 'Color_Şampanya', 'Body Type_-', 'Body Type_Cabrio', 'Body Type_Camlı Van',
            'Body Type_Coupe', 'Body Type_Crossover', 'Body Type_Frigorifik Panelvan', 'Body Type_Hard top',
            'Body Type_Hatchback/3', 'Body Type_Hatchback/5', 'Body Type_Kamyonet', 'Body Type_MPV',
            'Body Type_Minibüs', 'Body Type_Panel Van', 'Body Type_Pick-Up', 'Body Type_Pick-up',
            'Body Type_Roadster', 'Body Type_SUV', 'Body Type_Sedan', 'Body Type_Station wagon',
            'Body Type_Yarım Camlı Van', 'Drive_-', 'Drive_4WD (Sürekli)', 'Drive_4x2 (Arkadan İtişli)',
            'Drive_4x2 (Önden Çekişli)', 'Drive_4x4', 'Drive_AWD (Elektronik)', 'Drive_Arkadan İtiş',
            'Drive_Önden Çekiş']

    data = [0] * len(keys)

    for i in range(len(keys)):
        if keys[i] in user_data.keys():
            data[i] = user_data[keys[i]]

        else:
            for value in user_data.values():
                if '_' + str(value) in keys[i]:
                    data[i] = 1

    indices = X['Series'] == data[0]
    data[0] = int(np.mean(y[indices]))

    indices = X['Model'] == data[1]
    data[1] = int(np.mean(y[indices]))

    return np.array(data).astype(int)


app = Flask(__name__)

# Load the saved model
try:
    model = joblib.load('car_price_prediction_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

car_info, X, y = car_models()
car_brands = sorted(car_info.keys())
gear_type_info = sorted(['Düz', 'Otomatik', 'Yarı Otomatik'])
color_info = sorted(['Beyaz', 'Gri (Gümüş)', 'Mavi (metalik)', 'Mavi', 'Gri', 'Kırmızı', 'Füme', 'Yeşil', 'Siyah',
                     'Bej', 'Gri (metalik)', 'Yeşil (metalik)', 'Turkuaz', 'Gri (titanyum)', 'Sarı', 'Turuncu',
                     'Kahverengi', 'Lacivert', 'Bordo', 'Diğer', 'Şampanya', 'Mor', 'Altın', 'Pembe'])

fuel_type_info = sorted(['Benzin', 'Dizel', 'Elektrik', 'Hibrit', 'LPG & Benzin'])
body_type_info = sorted(['Sedan', 'Camlı Van', 'Crossover', 'Frigorifik Panelvan', 'Hatchback/5', 'Panel Van',
                         'Station wagon', 'Coupe', 'MPV', 'Hatchback/3', 'Minibüs', 'Hard top', 'SUV',
                         'Yarım Camlı Van', 'Cabrio', 'Pick-Up', 'Roadster', 'Kamyonet'])

drive_info = sorted(['4WD (Sürekli)', '4x2 (Arkadan İtişli)', '4x2 (Önden Çekişli)', '4x4', 'AWD (Elektronik)',
                     'Arkadan İtiş', 'Önden Çekiş'])


@app.route('/')
def home():
    return render_template('index.html', car_brands=car_brands,
                           dropdown_options=car_info,
                           gear_type_info=gear_type_info,
                           color_info=color_info,
                           fuel_type_info=fuel_type_info,
                           body_type_info=body_type_info,
                           drive_info=drive_info)


@app.route('/get_series_options', methods=['POST'])
def get_series_options():
    selected_model = request.json.get('selected_model')
    options = list(car_info.get(selected_model, {}).keys())
    return jsonify(options)


@app.route('/get_list_items', methods=['POST'])
def get_list_items():
    selected_model = request.json.get('selected_model')
    selected_series = request.json.get('selected_series')
    options = car_info.get(selected_model, {}).get(selected_series, [])
    return jsonify(options)


@app.route('/OneForAll_PricePredictor', methods=['POST'])
def predict():
    try:
        # Debugging: Print all received form data
        print("Form Data: ", request.form)

        # Get data from form using specific field names
        feature1 = request.form.get('feature1', type=float)
        feature2 = request.form.get('feature2', type=float)
        feature3 = request.form.get('feature3', type=float)
        feature4 = request.form.get('feature4', type=float)
        feature5 = request.form.get('feature5', type=float)
        feature6 = request.form.get('feature6', type=float)
        dropdown1 = request.form['dropdown1']
        dropdown2 = request.form['dropdown2']
        dropdown3 = request.form['dropdown3']
        dropdown4 = request.form['dropdown4']
        dropdown5 = request.form['dropdown5']
        dropdown6 = request.form['dropdown6']
        dropdown7 = request.form['dropdown7']
        dropdown8 = request.form['dropdown8']


        # Ensure all fields are received
        if (feature1 is None or feature2 is None or feature3 is None or
                feature4 is None or feature5 is None or feature6 is None or
                dropdown1 is None or dropdown2 is None or dropdown3 is None or
                dropdown4 is None or dropdown5 is None or dropdown6 is None or
                dropdown7 is None or dropdown8 is None):
            return render_template('index.html', prediction_text='Complete all areas.',
                                   car_brands=car_brands,
                                   dropdown_options=car_info,
                                   gear_type_info=gear_type_info,
                                   color_info=color_info,
                                   fuel_type_info=fuel_type_info,
                                   body_type_info=body_type_info,
                                   drive_info=drive_info
                                   )

        # Create a dict of features for the model
        user_data = {'Brand': dropdown1, 'Series': dropdown2, 'Model': dropdown3, 'Year': feature1,
                     'Gear Type': dropdown5, 'Color': dropdown6, 'Fuel Type': dropdown8, 'Kilometer': feature2,
                     'Engine Volume': feature3, 'Engine Power': feature4, 'Body Type': dropdown4, 'Drive': dropdown7,
                     'Paint-changed': feature5, 'Fuel Tank': feature6}

        #print(f"Data received: {user_data}")

        # Convert data into the format needed by the model
        final_features = correct_data(user_data, X, y)
        #print(f"Formatted features: {final_features}")

        # Make prediction
        prediction = model.predict([final_features])
        #print(f"Prediction: {prediction}")

        # Decode the prediction if necessary
        output = prediction[0]
        #print(f"Output: {output}")

        return render_template('index.html', prediction_text=f'Predicted Car Price: {output:.2f} TL',
                               car_brands=car_brands,
                               dropdown_options=car_info,
                               gear_type_info=gear_type_info,
                               color_info=color_info,
                               fuel_type_info=fuel_type_info,
                               body_type_info=body_type_info,
                               drive_info=drive_info
                               )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction.",
                               car_brands=car_brands,
                               dropdown_options=car_info,
                               gear_type_info=gear_type_info,
                               color_info=color_info,
                               fuel_type_info=fuel_type_info,
                               body_type_info=body_type_info,
                               drive_info=drive_info
                               )


if __name__ == "__main__":
    app.run()
