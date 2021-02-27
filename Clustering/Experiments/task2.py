#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
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

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 0, 0, 1]

for i in range(50):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X, y)
	
# 
# print("weights between input and first hidden layer:")
# print(clf.coefs_[0])
# print("\nweights between first hidden and second hidden layer:")
# print(clf.coefs_[1])

end = time.time()
duration = end - start
cpu_usage = myProcess.cpu_percent(interval=0)
memory_usage = myProcess.memory_percent()

print(duration)
f = open(sys.argv[1], "a")

print(duration)
f.write(str(duration) + "\n")
f.close()
