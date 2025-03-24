from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np

import pickle


def main():
    N = 1000000

    # Create sample dataset
    data = pd.DataFrame()
    data["P1"] = np.random.randint(0, 100, N)
    data["P2"] = np.random.randint(0, 100, N)
    data["P3"] = np.random.randint(0, 100, N)
    data["P4"] = np.random.randint(0, 100, N)
    data["P5"] = np.random.randint(0, 100, N)

    # Create a synthetic target variable
    data["T1"] = (
        1 * data["P1"]
        + 2 * data["P2"]
        + 3 * data["P3"]
        + 4 * data["P4"]
        + 5 * data["P5"]
        + np.random.normal(0, 10, N)
    )

    # Split the data into training and test sets
    X = data[["P1", "P2", "P3", "P4", "P5"]]
    y = data["T1"]

    data.to_csv("./datasets/sample-linear-dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Save the model
    with open("SampleLinearModel.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
