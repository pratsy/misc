from __future__ import print_function
from sklearn import tree
import numpy as np
import kaggle

x_train = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
y_train = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
x_test = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

depths = np.array([40,45,50,55,60])
folds = 5
min_err = 922337203685477580
min_fold = 0
min_depth = 0
print(x_train.shape)

"""
for folds in range(50, 100, 5):
#Dividing the training input and output into given number of folds
    x_train_folds = np.array_split(x_train, folds)
    y_train_folds = np.array_split(y_train, folds)
    print("---------")
#Iterating over different values of depth for the decision tree
    for d in depths:
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
            dt = tree.DecisionTreeRegressor(max_depth=d)
            dt = dt.fit(xx_train, yy_train)
            y_pred = dt.predict(xx_test)
            error += np.abs(y_pred - yy_test).mean()
            #error += run_me.compute_error(y_pred, YY_test)
        print(error/folds)
        mse = error/folds
        if (mse < min_err):
            min_err = mse
            min_depth = d
            min_fold = folds
            # print(min_err)
            # print(min_depth)
            # print(min_fold)
"""
dt = tree.DecisionTreeRegressor(max_depth=47)
dt = dt.fit(x_train, y_train)
predicted_y = dt.predict(x_test)
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)




