from __future__ import print_function
from sklearn import linear_model
import numpy as np
import kaggle

x_train = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
y_train = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
x_test = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

alpha = np.array([10**-4, 10**-2, 1, 10])
folds = 5

#Dividing the training input and output into given number of folds
x_train_folds = np.array_split(x_train, 5)
y_train_folds = np.array_split(y_train, 5)

#Iterating over different values of depth for the decision tree
for a in alpha:
    #Iterating over given number of folds
    error = 0.0
    for f in range(folds):
        xx_train = []
        yy_train = []
        xx_test = np.array(x_train_folds[f])
        yy_test = np.array(y_train_folds[f])
        for x in range(folds):
            if (x != f):
                xx_train.append(x_train_folds[x])
                yy_train.append(y_train_folds[x])
        xx_train = np.concatenate(np.array(xx_train), axis=0)
        yy_train = np.concatenate(np.array(yy_train), axis=0)
        reg = linear_model.Ridge(alpha=a)
        reg = reg.fit(xx_train, yy_train)
        y_pred = reg.predict(xx_test)
        error += np.abs(y_pred - yy_test).mean()
        #error += run_me.compute_error(y_pred, YY_test)
    print(error/folds)
reg = linear_model.Ridge(alpha=10**-2)
reg = reg.fit(x_train, y_train)
predicted_y = reg.predict(x_test)
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)



