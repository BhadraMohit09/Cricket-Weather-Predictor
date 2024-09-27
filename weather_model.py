import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the data from CSV
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def train_model(data):
    features = data[['Temperature', 'Humidity', 'WindSpeed']]
    target = data['Rain']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open('weather_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model

def predict_weather(model, temperature, humidity, windspeed):
    input_data = pd.DataFrame([[temperature, humidity, windspeed]], columns=['Temperature', 'Humidity', 'WindSpeed'])
    prediction = model.predict(input_data)
    return prediction

# Example usage:
if __name__ == "__main__":
    # Load the data
    data = load_data('data/weather_data.csv')

    model = train_model(data)
    
    # Example prediction
    result = predict_weather(model, 30, 65, 10)
    print(f"Predicted Rain Probability: {result[0]}")
