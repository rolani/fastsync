from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
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


X, y = make_regression(n_samples=20, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)
regr = MLPRegressor(random_state=1, max_iter=50).fit(X_train, y_train)
# regr.predict(X_test[:2])
# 
# regr.score(X_test, y_test)

end = time.time()
duration = end - start
cpu_usage = myProcess.cpu_percent(interval=0)
memory_usage = myProcess.memory_percent()

print(duration)
f = open(sys.argv[1], "a")

print(duration)
f.write(str(duration) + "\n")
f.close()
