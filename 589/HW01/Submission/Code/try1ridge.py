from __future__ import print_function
from sklearn import linear_model
import numpy as np
import kaggle

x_train = np.loadtxt('../../Data/PowerOutput/data_train.txt')
y_train = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
x_test = np.loadtxt('../../Data/PowerOutput/data_test.txt')

alpha = np.array([10-1,1,2,3,4,5,6,7,8,15,20,25])
#folds = 10
min_err = 922337203685477580
min_fold = 0
min_depth = 0
# print(x_train.shape)
# print(min(y_train))

for folds in range(5,200,5):
#Dividing the training input and output into given number of folds
    x_train_folds = np.array_split(x_train, folds)
    y_train_folds = np.array_split(y_train, folds)

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
        mse = error/folds
        #print(error/folds)
        if(mse < min_err):
            min_err = mse
            min_depth = a
            min_fold = folds
            print(min_err)
            print(min_depth)
            print(min_fold)

# dt = tree.DecisionTreeRegressor(max_depth=16)
# dt = dt.fit(x_train, y_train)
# predicted_y = dt.predict(x_test)
# file_name = '../Predictions/PowerOutput/best.csv'
# # Writing output in Kaggle format
# print('Writing output to ', file_name)
# kaggle.kaggleize(predicted_y, file_name)




