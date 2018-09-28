from __future__ import print_function
from knn_utils import KnnUtils
import numpy as np

indoor_dir = '././Data/IndoorLocalization'
power_plant_dir = 'C:/Users/Shruti1992/Desktop/589/HW01/Data/PowerOutput'

Y_filename = power_plant_dir + '/data_train.txt'
Y_filename = power_plant_dir + '/labels_train.txt'
test_filename = power_plant_dir + '/data_test.txt'
X_train = np.loadtxt(Y_filename)
Y_train = np.loadtxt(Y_filename)
X_test = np.loadtxt(test_filename)
k = 1
print(X_test.shape)
obj = KnnUtils()
obj.train(X_train, Y_train)
dist = obj.calculate_distance(X_test)
label = obj.predict(dist, k)
