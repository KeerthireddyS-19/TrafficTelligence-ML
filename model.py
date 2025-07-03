import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def preprocess_data(df):
    df = df.dropna()
    features = ['temp', 'rain', 'clouds', 'hour', 'holiday']
    target = 'traffic_volume'
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

def save_model(model, filename='traffic_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def run_pipeline():
    df = load_data("data/traffic_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
