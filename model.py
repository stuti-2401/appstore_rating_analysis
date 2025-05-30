import pandas as pd
import numpy as np
from data import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model():
    df = load_and_clean_data()
    df = df[['Installs', 'Price', 'Size_MB', 'Rating']].dropna()

    X = df[['Installs', 'Price', 'Size_MB']]
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse:.3f}")

    return model

if __name__ == "__main__":
    train_model()
