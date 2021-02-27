import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import datetime
import time
import pickle
import numpy
import os, psutil
import numpy as np
import sys

myProcess = psutil.Process(os.getpid())
start = time.time()
cpu_usage_base = myProcess.cpu_percent(interval=0)

# Location of dataset
url = "iris.data"

# Assign colum names to the dataset
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url)

# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])


# le = preprocessing.LabelEncoder()
# 
# y = y.apply(le.fit_transform)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=200)
mlp.fit(X_train, y_train.values.ravel())

#predictions = mlp.predict(X_test)

end = time.time()
duration = end - start
cpu_usage = myProcess.cpu_percent(interval=0)
memory_usage = myProcess.memory_percent()
f = open(sys.argv[1], "a")

print(duration)
f.write(str(duration) + "\n")
f.close()
