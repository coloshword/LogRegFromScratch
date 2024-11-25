import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## HYPERParameters
TEST_SIZE = 0.2

def clean_and_split_data(input_data, labels):
    
    # rain is 1, no rain is 0
    def normalize_rain_no_rain(s):
        if s == 'rain':
            return 1
        else:
            return 0

    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    # we also want to turn all yes and no's in the 0 and 1, respectively, making it float 32 
    labels = labels.apply(normalize_rain_no_rain)
    labels = labels.to_numpy().astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=TEST_SIZE, random_state=40)
    return X_train, X_test, y_train, y_test 

def eval(model, X_test, y_test):
    num_correct = 0
    for i in range(len(X_test)):
        # get prediction for X_test, see if its equal to y_test
        correct = y_test[i]
        pred = model.predict(X_test[i])
        if pred == correct:
            num_correct += 1
    return num_correct / len(y_test)

def translate_label(x):
    if x == 1.0:
        return 'rain'
    elif x == 0.0:
        return 'no rain'
    else:
        raise Exception("not valid label")

def main():
    df = pd.read_csv("./weather_forecast_data.csv")
    input_data = df.iloc[:, 0:5]
    labels = df.iloc[:, 5]
    X_train, X_test, y_train, y_test = clean_and_split_data(input_data, labels)
    log_reg = LogisticRegression(30000, 0.0005)
    log_reg.fit(X_train, y_train)
    print()
    print('--------------------------------------------------------')
    print("Finished Training")
    print(f"Evaluation {eval(log_reg, X_test, y_test)}")
    # run predict with a random example
    print("Testing with example: ", X_test[0])
    print("Correct label: ", translate_label(y_test[0]))
    print("Prediction: ", translate_label(log_reg.predict(X_test[0])))



if __name__ == "__main__":
    main()
