from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = {}
model['LG'] = LogisticRegression()
model['LDA'] = LinearDiscriminantAnalysis()
model['KNN'] = KNeighborsClassifier()
model['CART'] = DecisionTreeClassifier()
model['NB'] = GaussianNB()
model['SVM'] = SVC()
results = []
for key in model:
    result = cross_val_score(model[key], X, Y, cv=kfold, scoring='accuracy')
    results.append(result)
    print('%s: %.3f(%.3f)' % (key, result.mean(), result.std()))

#图表显示
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(model.keys())
plt.show()