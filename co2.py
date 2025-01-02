import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR,SVC 
from sklearn.metrics import mean_absolute_error,r2_score
def create_recursive_data(data,window_size=5):
    i=1
    while i< window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis =0)
    return data

data = pd.read_csv("co2.csv")
#print(data.info())
data["time"] = pd.to_datetime(data["time"])
# Filling missing values with interpolation
data["co2"] = data["co2"].interpolate()
"""
fig, ax = plt.subplots()
ax.plot(data["time"],data["co2"])
ax.set_xlabel("Time")
ax.set_ylabel("CO2")
plt.show()
"""
data = create_recursive_data(data,5)
#print(data)
x= data.drop(["time","target"],axis =1)
y = data["target"]
train_size = 0.8
num_samples = len (x)
x_train = x[:int(num_samples * train_size)]
y_train = y[:int(num_samples * train_size)]
x_test = x[int(num_samples * train_size):]
y_test = y[int(num_samples * train_size):]
reg = LinearRegression()    
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)
print(r2_score(y_test,y_predict))
print(mean_absolute_error(y_test,y_predict))
fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples * train_size)],data["co2"][:int(num_samples * train_size)],label = "train")
ax.plot(data["time"][int(num_samples * train_size):],data["co2"][int(num_samples * train_size):], label = "test")
ax.plot(data["time"][int(num_samples * train_size):],y_predict,label = "predict")
ax.set_xlabel("Time")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()