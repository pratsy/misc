from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
# Create a random forest classifier

x_train = np.loadtxt('../../Data/PowerOutput/data_train.txt')
y_train = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
x_test = np.loadtxt('../../Data/PowerOutput/data_test.txt')
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(x_train, y_train)

# Print the name and gini importance of each feature
print(clf.feature_importances_)