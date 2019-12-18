from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from pickle import dump
# from pickle import load
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

import warnings
warnings.filterwarnings('ignore')

filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]

test_size = 0.33
seed = 4
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression()
model.fit(X_train, y_train)

#保存模型
model_file = 'finalized_model.sav'
with open(model_file, 'wb') as model_f:
    #模型序列化
    dump(model, model_f)

#加载模型
with open(model_file, 'rb') as model_f:
    #模型反序列化
    loaded_model = load(model_f)
    result = loaded_model.score(X_test, y_test)
    print("算法评估结果： %.3f%%" % (result*100))